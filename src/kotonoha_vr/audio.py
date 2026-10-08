"""Windows endpoint audio capture and in-memory speech transcription.

This first version submits complete energy-segmented phrases, not word-by-word
streaming. WASAPI loopback captures an output endpoint's mixed audio; it does
not isolate a process, a VRChat player, or a person's identity. No recording or
transcript is written to disk. A local model may download its own model files.

Audio callbacks only collect/segment samples. User callbacks run on a separate,
bounded dispatcher and should enqueue recognition work rather than perform it.
``transcribe_clip`` is synchronous and belongs on the application's ASR worker.
"""

from __future__ import annotations

import importlib
import io
import math
import os
import queue
import re
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np


class AudioError(RuntimeError):
    """An actionable audio error, safe to show without credentials or audio."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool = False


@dataclass
class AudioClip:
    samples: np.ndarray
    sample_rate: int
    source: str

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise AudioError("音频片段必须是单声道数组。")
        if self.sample_rate <= 0 or int(self.sample_rate) != self.sample_rate:
            raise AudioError("音频采样率必须是正整数。")
        if not np.isfinite(samples).all():
            raise AudioError("音频片段包含无效采样值。")
        self.samples = np.array(samples, dtype=np.float32, copy=True, order="C")
        self.sample_rate = int(self.sample_rate)


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str | None
    # Detection probability when the backend supplies it, NOT word accuracy.
    confidence: float | None = None


def _load_pyaudio() -> Any:
    if sys.platform != "win32":
        raise AudioError("音频采集需要 Windows WASAPI；当前平台可使用文本演示。")
    try:
        return importlib.import_module("pyaudiowpatch")
    except (ImportError, OSError):
        raise AudioError("无法加载 PyAudioWPatch，请安装 Windows 音频依赖后重试。") from None


def list_devices() -> list[AudioDevice]:
    """List WASAPI microphones and the generated *input* loopback devices."""
    module = _load_pyaudio()
    pa = None
    try:
        pa = module.PyAudio()
        api_index = int(pa.get_host_api_info_by_type(module.paWASAPI)["index"])
        result = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            channels = int(info.get("maxInputChannels", 0))
            if int(info.get("hostApi", -1)) != api_index or channels <= 0:
                continue
            result.append(AudioDevice(
                index=index,
                name=str(info["name"]),
                channels=channels,
                sample_rate=int(info["defaultSampleRate"]),
                is_loopback=bool(info.get("isLoopbackDevice", False)),
            ))
        return result
    except AudioError:
        raise
    except Exception:
        raise AudioError("无法枚举 WASAPI 设备，请检查 Windows 音频服务及 VD 连接。") from None
    finally:
        if pa is not None:
            pa.terminate()


def _rms(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class _EnergySegmenter:
    """Deterministic phrase segmentation, also used by the hardware-free tests.

    Continuous input splits on a pause or the size limit. Push-to-talk input
    sets ``split_on_pause=False`` and ``split_on_max=False``; it keeps at most
    ``max_seconds`` and emits only when the caller explicitly calls ``flush``.
    """

    def __init__(self, sample_rate: int, threshold: float = 0.015,
                 pause_ms: int = 650, max_seconds: float = 12,
                 *, split_on_pause: bool = True, split_on_max: bool = True,
                 pre_roll_ms: int = 180, min_voice_ms: int = 80) -> None:
        if sample_rate <= 0 or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise AudioError("采样率或音量阈值无效。")
        if pause_ms <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise AudioError("停顿时间和录音上限必须大于零。")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.pause_samples = max(1, round(sample_rate * pause_ms / 1000))
        self.max_samples = max(1, round(sample_rate * max_seconds))
        self.frame_samples = max(1, round(sample_rate * 0.02))
        self.pre_samples = min(round(sample_rate * pre_roll_ms / 1000), self.max_samples // 4)
        self.min_voice = min(round(sample_rate * min_voice_ms / 1000), self.max_samples)
        self.tail_samples = round(sample_rate * 0.12)
        self.split_on_pause = split_on_pause
        self.split_on_max = split_on_max
        self.reset()

    def reset(self) -> None:
        self._pre: deque[np.ndarray] = deque()
        self._pre_size = 0
        self._parts: list[np.ndarray] = []
        self._size = 0
        self._voiced = 0
        self._trailing = 0
        self.capped = False

    def _remember(self, samples: np.ndarray) -> None:
        if not self.pre_samples:
            return
        self._pre.append(samples.copy())
        self._pre_size += samples.size
        while self._pre_size > self.pre_samples:
            excess = self._pre_size - self.pre_samples
            first = self._pre.popleft()
            if first.size > excess:
                self._pre.appendleft(first[excess:])
                self._pre_size -= excess
            else:
                self._pre_size -= first.size

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise AudioError("分句器需要单声道音频。")
        output: list[np.ndarray] = []
        offset = 0
        while offset < samples.size and not self.capped:
            frame = samples[offset:offset + self.frame_samples]
            voiced = _rms(frame) >= self.threshold
            if not self._parts:
                if not voiced:
                    self._remember(frame)
                    offset += frame.size
                    continue
                self._parts = list(self._pre)
                self._size = self._pre_size
                self._pre.clear()
                self._pre_size = 0
            # A hard cut may fall inside this frame. Process its remainder on
            # the next iteration so continuous speech is never silently lost.
            count = min(frame.size, self.max_samples - self._size)
            frame = frame[:count]
            self._parts.append(frame.copy())
            self._size += count
            self._voiced += count if voiced else 0
            self._trailing = 0 if voiced else self._trailing + count
            offset += count
            if self._size >= self.max_samples:
                if self.split_on_max:
                    clip = self.flush(trim_silence=False)
                    if clip is not None:
                        output.append(clip)
                else:
                    self.capped = True
            elif self.split_on_pause and self._trailing >= self.pause_samples:
                # Retain a little of the already silent tail as next pre-roll.
                pre = np.concatenate(self._parts)[-self.pre_samples:] if self.pre_samples else None
                clip = self.flush()
                if clip is not None:
                    output.append(clip)
                if pre is not None:
                    self._remember(pre)
        return output

    def flush(self, *, trim_silence: bool = True) -> np.ndarray | None:
        result = None
        if self._parts and self._voiced >= self.min_voice:
            end = self._size
            if trim_silence:
                end -= max(0, self._trailing - self.tail_samples)
            result = np.concatenate(self._parts)[:end].astype(np.float32, copy=False)
        self.reset()
        return result


class AudioCapture:
    """Capture a selected WASAPI input/loopback device without transcribing it.

    ``source='mic'`` starts disabled. Hold with ``set_enabled(True)`` and release
    with ``set_enabled(False)``; release flushes one clip. Other sources start
    enabled and are segmented continuously. ``stop()`` discards an unfinished
    phrase and queued callbacks; it never submits an accidental final utterance.
    Callback consumers must also reject stale ASR results after a UI mode change.
    This class controls capture only; it does not mute/unmute VRChat.
    """

    def __init__(self, device_index: int, source: str,
                 on_clip: Callable[[AudioClip], None],
                 on_level: Callable[[float], None] | None = None,
                 on_error: Callable[[str], None] | None = None,
                 threshold: float = 0.015, pause_ms: int = 650,
                 max_seconds: float = 12) -> None:
        # Validate settings before touching a driver.
        _EnergySegmenter(16000, threshold, pause_ms, max_seconds)
        self.device_index = device_index
        self.source = source
        self.on_clip = on_clip
        self.on_level = on_level
        self.on_error = on_error
        self.threshold = threshold
        self.pause_ms = pause_ms
        self.max_seconds = max_seconds
        self._mic = source == "mic"
        self._enabled = not self._mic
        self._lock = threading.RLock()
        self._lifecycle = threading.RLock()
        self._running = False
        self._generation = 0
        self._epoch = 0
        self._pa: Any = None
        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._segmenter: _EnergySegmenter | None = None
        self._events: queue.Queue[tuple[int, int, float, str, Any]] = queue.Queue(maxsize=8)
        self._latest_level = 0.0
        self._rate = 16000
        self._channels = 1

    def start(self) -> None:
        with self._lifecycle:
            if self._running:
                return
            module = _load_pyaudio()
            try:
                self._pa = module.PyAudio()
                info = self._pa.get_device_info_by_index(self.device_index)
                api = self._pa.get_host_api_info_by_index(int(info["hostApi"]))
                if int(api["type"]) != module.paWASAPI:
                    raise AudioError("请选择 Windows WASAPI 麦克风或回环设备。")
                self._channels = int(info.get("maxInputChannels", 0))
                self._rate = int(info["defaultSampleRate"])
                if self._channels < 1:
                    raise AudioError("该设备不能录音；接收字幕请选对应的 Loopback 回环设备。")
                if self._mic and bool(info.get("isLoopbackDevice", False)):
                    raise AudioError("中文准备需要选择麦克风，不能选择扬声器回环。")
                with self._lock:
                    self._generation += 1
                    run = self._generation
                    self._events = queue.Queue(maxsize=8)
                    self._segmenter = _EnergySegmenter(
                        self._rate, self.threshold, self.pause_ms, self.max_seconds,
                        split_on_pause=not self._mic, split_on_max=not self._mic,
                        pre_roll_ms=0 if self._mic else 180,
                    )
                    self._latest_level = 0.0
                    self._running = True
                self._stream = self._pa.open(
                    format=module.paInt16, channels=self._channels, rate=self._rate,
                    input=True, input_device_index=self.device_index,
                    frames_per_buffer=max(1, round(self._rate * 0.02)), start=False,
                    stream_callback=lambda data, frames, clock, status: self._callback(
                        run, module, data, frames, clock, status),
                )
                self._thread = threading.Thread(
                    target=self._dispatch, args=(run,),
                    name=f"audio-dispatch-{self.source}", daemon=True,
                )
                self._thread.start()
                self._stream.start_stream()
            except AudioError:
                self.stop()
                raise
            except Exception:
                self.stop()
                raise AudioError("无法打开音频设备，请检查 VD 连接、设备选择及独占模式。") from None

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            enabled = bool(enabled)
            if enabled == self._enabled:
                return
            self._epoch += 1
            self._enabled = enabled
            if self._segmenter is None:
                return
            if self._mic and not enabled and self._running:
                clip = self._segmenter.flush()
                if clip is not None:
                    self._enqueue("clip", AudioClip(clip, self._rate, self.source))
            else:
                self._segmenter.reset()
            self._latest_level = 0.0

    def stop(self) -> None:
        self._stop()

    def _stop(self, expected_generation: int | None = None) -> None:
        with self._lifecycle:
            with self._lock:
                if expected_generation is not None and expected_generation != self._generation:
                    return
                self._running = False
                self._generation += 1
                self._epoch += 1
                if self._mic:
                    self._enabled = False
                if self._segmenter is not None:
                    self._segmenter.reset()
                self._latest_level = 0.0
                stream, pa, thread = self._stream, self._pa, self._thread
                self._stream = self._pa = self._thread = None
                while not self._events.empty():
                    try:
                        self._events.get_nowait()
                    except queue.Empty:
                        break
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
        if thread is not None and thread is not threading.current_thread():
            # Release the lifecycle lock before joining: a failing dispatcher's
            # cleanup may be waiting for it. Its generation guard prevents that
            # old cleanup from stopping a newly started stream.
            thread.join(timeout=2.0)

    def _enqueue(self, kind: str, value: Any) -> None:
        event = (self._generation, self._epoch, time.monotonic(), kind, value)
        try:
            self._events.put_nowait(event)
        except queue.Full:
            # Bound memory and prefer recent speech if a consumer is too slow.
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(event)
            except queue.Full:
                pass

    def _callback(self, run: int, module: Any, data: bytes,
                  frames: int, clock: Any, status: int) -> tuple[None, int]:
        del frames, clock
        try:
            with self._lock:
                if not self._running or run != self._generation:
                    return None, module.paComplete
                if not self._enabled:
                    self._latest_level = 0.0
                    return None, module.paContinue
                if status & getattr(module, "paInputOverflow", 2):
                    # A gap must not glue two unrelated fragments together.
                    self._segmenter.reset()
                    self._enqueue("error", "音频缓冲发生丢帧，已丢弃未完成片段。")
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                if self._channels > 1:
                    samples = samples.reshape(-1, self._channels).mean(axis=1, dtype=np.float32)
                self._latest_level = min(1.0, _rms(samples))
                was_capped = self._segmenter.capped
                for part in self._segmenter.feed(samples):
                    self._enqueue("clip", AudioClip(part, self._rate, self.source))
                if self._segmenter.capped and not was_capped:
                    self._enqueue("error", f"本次准备已达 {self.max_seconds:g} 秒，请松开按键；仅保留上限内的语音。")
            return None, module.paContinue
        except Exception:
            with self._lock:
                if self._running and run == self._generation:
                    self._enqueue("fatal", "音频采集中断，请检查设备连接后重新开始。")
            return None, module.paAbort

    def _dispatch(self, run: int) -> None:
        last_level = 0.0
        while True:
            with self._lock:
                if not self._running or run != self._generation:
                    return
                events = self._events
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                event = None
            if event is not None:
                generation, epoch, created, kind, value = event
                with self._lock:
                    valid = (self._running and generation == self._generation
                             and (epoch == self._epoch or kind == "fatal"))
                if valid and time.monotonic() - created <= max(30.0, self.max_seconds * 2):
                    callback = self.on_clip if kind == "clip" else self.on_error
                    if callback is not None:
                        try:
                            callback(value)
                        except Exception:
                            # UI failures must never escape to the audio driver.
                            pass
                    if kind == "fatal":
                        self._stop(expected_generation=run)
                        return
            now = time.monotonic()
            if now - last_level >= 0.1:
                with self._lock:
                    if not self._running or run != self._generation:
                        return
                    level = self._latest_level
                if self.on_level is not None:
                    try:
                        self.on_level(level)
                    except Exception:
                        pass
                last_level = now


_LANGUAGE_NAMES = {
    "english": "en", "chinese": "zh", "mandarin": "zh", "mandarin chinese": "zh",
    "cantonese": "yue", "japanese": "ja", "korean": "ko", "spanish": "es",
    "french": "fr", "german": "de", "portuguese": "pt", "russian": "ru",
    "italian": "it", "arabic": "ar", "hindi": "hi", "dutch": "nl",
    "turkish": "tr", "polish": "pl", "swedish": "sv", "indonesian": "id",
    "vietnamese": "vi", "thai": "th", "hebrew": "he", "ukrainian": "uk",
    "greek": "el", "czech": "cs", "romanian": "ro", "danish": "da",
    "finnish": "fi", "norwegian": "no", "hungarian": "hu", "persian": "fa",
    "farsi": "fa", "tamil": "ta", "telugu": "te", "malay": "ms",
    "bengali": "bn", "slovak": "sk", "catalan": "ca", "croatian": "hr",
    "serbian": "sr", "bulgarian": "bg", "slovenian": "sl", "urdu": "ur",
    "lithuanian": "lt", "latvian": "lv", "estonian": "et", "icelandic": "is",
    "filipino": "tl", "tagalog": "tl", "afrikaans": "af", "swahili": "sw",
    "albanian": "sq", "amharic": "am", "armenian": "hy", "assamese": "as",
    "azerbaijani": "az", "bashkir": "ba", "basque": "eu", "belarusian": "be",
    "bosnian": "bs", "breton": "br", "burmese": "my", "faroese": "fo",
    "galician": "gl", "georgian": "ka", "gujarati": "gu", "haitian creole": "ht",
    "hausa": "ha", "hawaiian": "haw", "javanese": "jw", "kannada": "kn",
    "kazakh": "kk", "khmer": "km", "lao": "lo", "latin": "la",
    "lingala": "ln", "luxembourgish": "lb", "macedonian": "mk", "malagasy": "mg",
    "malayalam": "ml", "maltese": "mt", "maori": "mi", "marathi": "mr",
    "mongolian": "mn", "nepali": "ne", "nynorsk": "nn", "occitan": "oc",
    "pashto": "ps", "punjabi": "pa", "sanskrit": "sa", "shona": "sn",
    "sindhi": "sd", "sinhala": "si", "somali": "so", "sundanese": "su",
    "tajik": "tg", "tatar": "tt", "tibetan": "bo", "turkmen": "tk",
    "uzbek": "uz", "welsh": "cy", "yiddish": "yi", "yoruba": "yo",
}
_LANGUAGE_ALIASES = {
    "eng": "en", "cmn": "zh", "zho": "zh", "chi": "zh", "jpn": "ja",
    "kor": "ko", "spa": "es", "fra": "fr", "fre": "fr", "deu": "de",
    "ger": "de", "por": "pt", "rus": "ru", "ita": "it", "ara": "ar",
    "hin": "hi", "nld": "nl", "dut": "nl", "vie": "vi", "tha": "th",
    "jv": "jw", "iw": "he", "中文": "zh", "英语": "en", "日语": "ja",
    "韩语": "ko", "chinese (simplified)": "zh", "chinese (traditional)": "zh",
}


def normalize_language(value: str | None) -> str | None:
    """Normalize detector names/locale codes; unknown is not presumed English."""
    if not isinstance(value, str):
        return None
    name = value.strip().lower().replace("_", "-")
    if name in {"", "auto", "unknown", "und", "mixed", "multilingual", "none"}:
        return None
    if name in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[name]
    if name in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[name]
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", name):
        code = name.split("-", 1)[0]
        return _LANGUAGE_ALIASES.get(code, code)
    return None


def _wav_bytes(clip: AudioClip) -> bytes:
    buffer = io.BytesIO()
    pcm = np.rint(np.clip(clip.samples, -1, 1) * 32767).astype("<i2")
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(clip.sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _probability(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and 0 <= value <= 1:
            return float(value)
    return None


def _parse_transcription(payload: Any) -> Transcription:
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise AudioError("语音服务返回了无法识别的结果格式。")
    languages: set[str] = set()
    language = normalize_language(payload.get("language"))
    if language is not None:
        languages.add(language)
    detected = payload.get("languages", [])
    if isinstance(detected, list):
        for item in detected:
            raw = item.get("code", item.get("language")) if isinstance(item, dict) else item
            code = normalize_language(raw)
            if code is not None:
                languages.add(code)
    # The single-language application contract cannot represent a mixed turn.
    # Return unknown so the caller can ask instead of choosing an arbitrary one.
    language = next(iter(languages)) if len(languages) == 1 else None
    confidence = _probability(payload.get("language_probability")) if language else None
    return Transcription(payload["text"].strip(), language, confidence)


_MODEL_LOCK = threading.Lock()
_LOCAL_MODEL: tuple[str, Any] | None = None


def _transcribe_local(clip: AudioClip, model: str, language_hint: str | None) -> Transcription:
    global _LOCAL_MODEL
    with _MODEL_LOCK:
        if _LOCAL_MODEL is None or _LOCAL_MODEL[0] != model:
            try:
                module = importlib.import_module("faster_whisper")
            except (ImportError, OSError):
                raise AudioError("本地识别需要 faster-whisper，请先安装本地语音依赖。") from None
            # Only one shared CPU model stays cached; all inference is serial.
            _LOCAL_MODEL = None
            try:
                loaded = module.WhisperModel(
                    model, device="cpu", compute_type="int8",
                    cpu_threads=max(1, min(4, os.cpu_count() or 2)), num_workers=1,
                )
                _LOCAL_MODEL = (model, loaded)
            except Exception:
                raise AudioError("本地模型加载失败；首次使用需下载模型，也可填写已下载模型目录。") from None
        try:
            # A raw ndarray is assumed to be 16 kHz by faster-whisper. Supplying
            # a WAV BytesIO lets its decoder resample the actual device rate.
            segments, info = _LOCAL_MODEL[1].transcribe(
                io.BytesIO(_wav_bytes(clip)), task="transcribe", language=language_hint,
                beam_size=1, vad_filter=True, condition_on_previous_text=False,
            )
            # The generator performs inference: keep the lock until exhausted.
            text = "".join(segment.text for segment in segments).strip()
            detected = normalize_language(getattr(info, "language", None))
            confidence = None if language_hint else _probability(getattr(info, "language_probability", None))
            return Transcription(text, detected, confidence)
        except Exception:
            raise AudioError("本地语音识别失败，请检查模型和音频输入后重试。") from None


def _transcribe_openai(clip: AudioClip, model: str, api_key: str,
                       base_url: str, language_hint: str | None) -> Transcription:
    try:
        httpx = importlib.import_module("httpx")
    except ImportError:
        raise AudioError("在线语音识别需要 httpx，请先安装应用依赖。") from None
    try:
        parsed = urlsplit(base_url.strip())
    except (ValueError, TypeError):
        raise AudioError("语音服务地址无效，请检查 API 根地址。") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise AudioError("语音服务地址无效，请填写不含账号、查询参数的 HTTP(S) API 根地址。")
    if not api_key.strip() and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise AudioError("请填写语音识别服务的 API Key。")
    if model in {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo", ""}:
        model = "whisper-1"
    is_new_transcribe = model == "gpt-transcribe" or model.startswith("gpt-transcribe-")
    data = {"model": model, "response_format": "verbose_json" if model == "whisper-1" else "json"}
    if language_hint:
        data["languages[]" if is_new_transcribe else "language"] = language_hint
    headers = {"Authorization": f"Bearer {api_key.strip()}"} if api_key.strip() else {}
    endpoint = base_url.strip().rstrip("/") + "/audio/transcriptions"
    try:
        # No automatic retries: a retry could duplicate a paid transcription.
        # Redirects stay disabled, so credentials cannot follow a new origin.
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=False) as client:
            response = client.post(
                endpoint, headers=headers, data=data,
                files={"file": ("speech.wav", _wav_bytes(clip), "audio/wav")},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        # Never display server bodies, request URLs, headers, or exception text:
        # compatible providers sometimes echo credentials in their responses.
        code = exc.response.status_code
        if code in {401, 403}:
            message = "语音服务拒绝认证，请检查 API Key 和模型访问权限。"
        elif code == 429:
            message = "语音服务额度不足或请求过多，请稍后重试。"
        else:
            message = f"语音服务请求失败（HTTP {code}），请检查服务地址及模型。"
        raise AudioError(message) from None
    except httpx.TimeoutException:
        raise AudioError("语音识别请求超时，请检查连接或切换本地识别。") from None
    except httpx.RequestError:
        raise AudioError("无法连接语音识别服务，请检查网络和服务地址。") from None
    except (ValueError, TypeError):
        raise AudioError("语音服务返回了无效响应。") from None
    except Exception:
        raise AudioError("语音请求未能完成，请检查服务地址、音频与连接设置。") from None
    return _parse_transcription(payload)


def transcribe_clip(clip: AudioClip, backend: str = "local", model: str = "small",
                    api_key: str = "", base_url: str = "https://api.openai.com/v1",
                    language_hint: str | None = None) -> Transcription:
    """Transcribe the original language, without translating or saving speech.

    Incoming speaker audio defaults to language detection. A Chinese preparation
    clip (``source='mic'``) defaults to the ``zh`` hint; explicitly pass ``auto``
    for detection instead. The owner must discard results of outdated requests;
    already-running synchronous local inference cannot be forcibly cancelled.
    """
    if not clip.samples.size or _rms(clip.samples) < 0.00001:
        return Transcription("", None)
    hint = normalize_language(language_hint if language_hint is not None else ("zh" if clip.source == "mic" else None))
    backend = backend.strip().lower()
    model = model.strip()
    if backend == "local":
        return _transcribe_local(clip, model or "small", hint)
    if backend == "openai":
        return _transcribe_openai(clip, model, api_key, base_url, hint)
    raise AudioError("未知语音识别后端，请选择 local 或 openai。")
