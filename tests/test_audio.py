"""Hardware-free audio tests. No model downloads, recording, or API calls."""

import io
import sys
import threading
import types
import unittest
import wave
from unittest import mock

import numpy as np

from kotonoha_vr import audio


def tone(seconds, rate=1000, volume=0.12):
    return np.full(round(seconds * rate), volume, dtype=np.float32)


class EnergySegmentationTests(unittest.TestCase):
    def test_silence_never_emits_a_clip(self):
        segmenter = audio._EnergySegmenter(1000, max_seconds=1)
        self.assertEqual(segmenter.feed(tone(5, volume=0)), [])
        self.assertIsNone(segmenter.flush())

    def test_pause_emits_with_preroll_and_short_tail(self):
        segmenter = audio._EnergySegmenter(1000, pause_ms=300)
        self.assertEqual(segmenter.feed(tone(0.4, volume=0)), [])
        self.assertEqual(segmenter.feed(tone(0.4)), [])
        clips = segmenter.feed(tone(0.4, volume=0))
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].dtype, np.float32)
        self.assertEqual(len(clips[0]), 180 + 400 + 120)
        np.testing.assert_array_equal(clips[0][:180], 0)
        np.testing.assert_allclose(clips[0][180:580], 0.12)
        self.assertIsNone(segmenter.flush())

    def test_hard_limits_do_not_lose_continuous_speech(self):
        segmenter = audio._EnergySegmenter(1000, max_seconds=0.255)
        original = tone(1.17)
        clips = segmenter.feed(original)
        tail = segmenter.flush()
        if tail is not None:
            clips.append(tail)
        self.assertEqual([len(clip) for clip in clips], [255, 255, 255, 255, 150])
        np.testing.assert_array_equal(np.concatenate(clips), original)

    def test_explicit_end_flush_preserves_unpaused_speech_once(self):
        segmenter = audio._EnergySegmenter(1000)
        self.assertEqual(segmenter.feed(tone(0.3)), [])
        self.assertEqual(len(segmenter.flush()), 300)
        self.assertIsNone(segmenter.flush())

    def test_mic_waits_for_release_despite_a_long_internal_pause(self):
        segmenter = audio._EnergySegmenter(
            1000, pause_ms=200, split_on_pause=False, split_on_max=False, pre_roll_ms=0)
        speech = np.concatenate([tone(0.4), tone(0.8, volume=0), tone(0.4)])
        self.assertEqual(segmenter.feed(speech), [])
        np.testing.assert_array_equal(segmenter.flush(), speech)

    def test_mic_limit_truncates_until_release(self):
        segmenter = audio._EnergySegmenter(
            1000, max_seconds=1, split_on_pause=False, split_on_max=False, pre_roll_ms=0)
        self.assertEqual(segmenter.feed(tone(1.5)), [])
        self.assertTrue(segmenter.capped)
        self.assertEqual(segmenter.feed(tone(1)), [])
        self.assertEqual(len(segmenter.flush()), 1000)
        self.assertFalse(segmenter.capped)

    def test_reset_discards_a_partial_phrase_and_preroll(self):
        segmenter = audio._EnergySegmenter(1000)
        segmenter.feed(tone(0.4))
        segmenter.reset()
        self.assertIsNone(segmenter.flush())
        self.assertEqual(segmenter.feed(tone(1, volume=0)), [])

    def test_tiny_noise_click_does_not_become_a_phrase(self):
        segmenter = audio._EnergySegmenter(1000)
        segmenter.feed(tone(0.02))
        self.assertEqual(segmenter.feed(tone(1, volume=0)), [])

    def test_clip_owns_valid_mono_float32_samples(self):
        original = tone(0.2)
        clip = audio.AudioClip(original, 48000, "speaker")
        original[:] = 0
        self.assertTrue(np.all(clip.samples != 0))
        for samples, rate in [(np.ones((3, 2)), 48000), (np.ones(3), 0), (np.array([np.nan]), 1)]:
            with self.assertRaises(audio.AudioError):
                audio.AudioClip(samples, rate, "speaker")


class _FakeStream:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        self.started = False

    def close(self):
        self.closed = True

    def inject(self, values, status=0):
        channels = self.kwargs["channels"]
        values = np.asarray(values, dtype=np.float32)
        if channels > 1 and values.ndim == 1:
            values = np.repeat(values[:, None], channels, axis=1)
        raw = np.rint(values * 32767).astype("<i2").tobytes()
        return self.kwargs["stream_callback"](raw, len(values), {}, status)


class _FakeAudio:
    def __init__(self):
        self.terminated = False
        self.stream = None
        self.devices = [
            {"name": "Speakers", "hostApi": 7, "maxInputChannels": 0, "defaultSampleRate": 48000},
            {"name": "Speakers [Loopback]", "hostApi": 7, "maxInputChannels": 2,
             "defaultSampleRate": 48000, "isLoopbackDevice": True},
            {"name": "VD microphone", "hostApi": 7, "maxInputChannels": 1, "defaultSampleRate": 48000},
            {"name": "Other API microphone", "hostApi": 1, "maxInputChannels": 1, "defaultSampleRate": 44100},
        ]

    def get_host_api_info_by_type(self, api_type):
        return {"index": 7, "type": 13}

    def get_host_api_info_by_index(self, index):
        return {"index": index, "type": 13 if index == 7 else 1}

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        return self.devices[index]

    def open(self, **kwargs):
        self.stream = _FakeStream(kwargs)
        return self.stream

    def terminate(self):
        self.terminated = True


class CaptureLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.instances = []

        def factory():
            instance = _FakeAudio()
            self.instances.append(instance)
            return instance

        self.module = types.SimpleNamespace(
            PyAudio=factory, paWASAPI=13, paInt16=8, paContinue=0,
            paComplete=1, paAbort=2, paInputOverflow=2,
        )
        self.patch = mock.patch.object(audio, "_load_pyaudio", return_value=self.module)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def capture(self, source="mic", **kwargs):
        self.clips = []
        self.ready = threading.Event()
        self.callback_thread = None

        def receive(clip):
            self.callback_thread = threading.current_thread().name
            self.clips.append(clip)
            self.ready.set()

        cap = audio.AudioCapture(2 if source == "mic" else 1, source, receive, **kwargs)
        self.addCleanup(cap.stop)
        cap.start()
        return cap, self.instances[-1].stream

    def test_enumerates_only_wasapi_inputs_including_loopback(self):
        devices = audio.list_devices()
        self.assertEqual([dev.index for dev in devices], [1, 2])
        self.assertTrue(devices[0].is_loopback)
        self.assertFalse(devices[1].is_loopback)
        self.assertTrue(self.instances[-1].terminated)

    def test_mic_is_disabled_then_flushes_on_release_off_capture_thread(self):
        cap, stream = self.capture()
        stream.inject(tone(0.4, rate=48000))
        cap.set_enabled(True)
        stream.inject(tone(0.4, rate=48000))
        stream.inject(tone(1, rate=48000, volume=0))
        self.assertFalse(self.ready.is_set())
        cap.set_enabled(False)
        self.assertTrue(self.ready.wait(1))
        self.assertEqual(len(self.clips), 1)
        self.assertEqual(self.clips[0].source, "mic")
        self.assertEqual(self.clips[0].sample_rate, 48000)
        self.assertEqual(len(self.clips[0].samples), round(48000 * 0.52))
        self.assertTrue(self.callback_thread.startswith("audio-dispatch-"))

    def test_speaker_downmixes_and_splits_on_pause(self):
        cap, stream = self.capture("speaker", pause_ms=200)
        speech = np.column_stack([tone(0.3, 48000, 0.2), tone(0.3, 48000, 0.0)])
        stream.inject(speech)
        stream.inject(tone(0.3, 48000, 0))
        self.assertTrue(self.ready.wait(1))
        clip = self.clips[0]
        self.assertEqual(clip.samples.ndim, 1)
        np.testing.assert_allclose(clip.samples[:100], 0.1, atol=0.0001)

    def test_stop_discards_unfinished_phrase_and_rejects_old_driver_callback(self):
        cap, stream = self.capture()
        cap.set_enabled(True)
        stream.inject(tone(0.4, rate=48000))
        cap.stop()
        cap.set_enabled(False)
        result = stream.inject(tone(0.4, rate=48000))
        self.assertEqual(result, (None, self.module.paComplete))
        self.assertFalse(self.ready.wait(0.1))
        self.assertTrue(stream.closed)
        self.assertTrue(self.instances[-1].terminated)

    def test_start_failure_releases_portaudio(self):
        cap = audio.AudioCapture(0, "speaker", lambda clip: None)
        with self.assertRaisesRegex(audio.AudioError, "Loopback"):
            cap.start()
        self.assertTrue(self.instances[-1].terminated)
        self.assertFalse(cap._running)

    def test_old_failure_cleanup_cannot_stop_a_restarted_capture(self):
        cap, old_stream = self.capture()
        old_generation = cap._generation
        cap.stop()
        cap.start()
        cap._stop(expected_generation=old_generation)
        self.assertTrue(cap._running)
        self.assertTrue(old_stream.closed)
        self.assertFalse(self.instances[-1].terminated)


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.clip = audio.AudioClip(tone(0.3, rate=48000), 48000, "speaker")

    def fake_http(self, payload=None, error=None):
        class HTTPStatusError(Exception):
            def __init__(self, status):
                self.response = types.SimpleNamespace(status_code=status)
                super().__init__("server echoed sk-secret-test-token")

        class RequestError(Exception):
            pass

        class TimeoutException(RequestError):
            pass

        response = mock.Mock()
        response.json.return_value = payload
        if error is not None:
            response.raise_for_status.side_effect = HTTPStatusError(error)
        client = mock.MagicMock()
        client.__enter__.return_value = client
        client.post.return_value = response
        module = types.SimpleNamespace(
            Client=mock.Mock(return_value=client), Timeout=mock.Mock(),
            HTTPStatusError=HTTPStatusError, RequestError=RequestError,
            TimeoutException=TimeoutException,
        )
        patch = mock.patch.dict(sys.modules, {"httpx": module})
        patch.start()
        self.addCleanup(patch.stop)
        return client, module

    def test_windows_dependency_is_lazy_and_other_platform_has_clear_error(self):
        with mock.patch.object(audio.sys, "platform", "linux"):
            with self.assertRaisesRegex(audio.AudioError, "Windows"):
                audio._load_pyaudio()

    def test_silence_does_not_load_any_model_or_call_a_service(self):
        clip = audio.AudioClip(tone(1, volume=0), 1000, "speaker")
        with mock.patch.object(audio, "_transcribe_local") as local:
            result = audio.transcribe_clip(clip)
        self.assertEqual(result, audio.Transcription("", None))
        local.assert_not_called()

    def test_whisper_verbose_json_preserves_text_and_normalizes_detected_name(self):
        client, module = self.fake_http({"text": " Bonjour ! ", "language": "french"})
        result = audio.transcribe_clip(self.clip, backend="openai", api_key="sk-secret-test-token")
        self.assertEqual(result, audio.Transcription("Bonjour !", "fr"))
        args, kwargs = client.post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/audio/transcriptions")
        self.assertEqual(kwargs["data"], {"model": "whisper-1", "response_format": "verbose_json"})
        filename, contents, mime = kwargs["files"]["file"]
        self.assertEqual((filename, mime), ("speech.wav", "audio/wav"))
        with wave.open(io.BytesIO(contents), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (48000, 1, 2))
            self.assertEqual(wav.getnframes(), len(self.clip.samples))
        self.assertFalse(module.Client.call_args.kwargs["follow_redirects"])

    def test_gpt_transcribe_extracts_language_codes(self):
        client, _ = self.fake_http({"text": "こんにちは", "languages": [{"code": "ja"}]})
        result = audio.transcribe_clip(self.clip, "openai", "gpt-transcribe", "test-key")
        self.assertEqual((result.text, result.language), ("こんにちは", "ja"))
        self.assertEqual(client.post.call_args.kwargs["data"], {"model": "gpt-transcribe", "response_format": "json"})

    def test_gpt_microphone_uses_plural_language_hint(self):
        client, _ = self.fake_http({"text": "我想一起去", "languages": [{"code": "zh-cn"}]})
        clip = audio.AudioClip(self.clip.samples, self.clip.sample_rate, "mic")
        result = audio.transcribe_clip(clip, "openai", "gpt-transcribe", "test-key")
        data = client.post.call_args.kwargs["data"]
        self.assertEqual(data["languages[]"], "zh")
        self.assertNotIn("language", data)
        self.assertEqual(result.language, "zh")

    def test_missing_or_multiple_languages_are_not_invented(self):
        for payload in [
            {"text": "hello"}, {"text": "hello", "languages": []},
            {"text": "hello bonjour", "languages": [{"code": "en"}, {"code": "fr"}]},
        ]:
            self.assertIsNone(audio._parse_transcription(payload).language)

    def test_server_errors_do_not_expose_api_keys_or_server_messages(self):
        self.fake_http(error=401)
        with self.assertRaises(audio.AudioError) as raised:
            audio.transcribe_clip(self.clip, "openai", "whisper-1", "sk-secret-test-token")
        self.assertNotIn("sk-secret", str(raised.exception))
        self.assertIn("认证", str(raised.exception))

    def test_invalid_url_is_sanitized_before_any_network_call(self):
        client, _ = self.fake_http()
        for address in ["https://[invalid", "https://example.test/v1?key=sk-secret-test-token"]:
            with self.assertRaises(audio.AudioError) as raised:
                audio.transcribe_clip(self.clip, "openai", "whisper-1", "test-key", address)
            self.assertNotIn("sk-secret", str(raised.exception))
        client.post.assert_not_called()

    def test_local_cache_is_shared_and_lock_covers_segment_generator(self):
        loaded = mock.Mock()

        def recognize(source, **kwargs):
            self.assertTrue(audio._MODEL_LOCK.locked())
            self.assertEqual(kwargs["task"], "transcribe")
            with wave.open(source, "rb") as wav:
                self.assertEqual(wav.getframerate(), 48000)

            def segments():
                self.assertTrue(audio._MODEL_LOCK.locked())
                yield types.SimpleNamespace(text=" こんにちは")

            return segments(), types.SimpleNamespace(language="ja", language_probability=0.93)

        loaded.transcribe.side_effect = recognize
        module = types.SimpleNamespace(WhisperModel=mock.Mock(return_value=loaded))
        with mock.patch.dict(sys.modules, {"faster_whisper": module}), mock.patch.object(audio, "_LOCAL_MODEL", None):
            first = audio.transcribe_clip(self.clip, "local", "small")
            second = audio.transcribe_clip(self.clip, "local", "small")
        self.assertEqual(first, second)
        self.assertEqual((first.language, first.confidence), ("ja", 0.93))
        self.assertEqual(module.WhisperModel.call_count, 1)
        self.assertEqual(module.WhisperModel.call_args.kwargs["device"], "cpu")
        self.assertEqual(module.WhisperModel.call_args.kwargs["compute_type"], "int8")

    def test_name_locale_and_iso_alias_normalization(self):
        for raw, expected in [
            ("English", "en"), ("Japanese", "ja"), ("Chinese", "zh"), ("Spanish", "es"),
            ("zh-TW", "zh"), ("eng", "en"), ("pt_BR", "pt"), ("Cantonese", "yue"),
            ("unknown", None), (None, None), ("not a language", None),
        ]:
            self.assertEqual(audio.normalize_language(raw), expected)


if __name__ == "__main__":
    unittest.main()
