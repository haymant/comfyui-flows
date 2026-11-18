from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"comfy_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


asr = load("asr")
qa = load("qa_audio")


RECORD = {
    "text": "Rates stay restrictive until inflation cools.",
    "language": "en",
    "model": "mlx-community/whisper-large-v3-turbo-asr-fp16",
    "backend": "mlx",
    "duration": 4.0,
    "segments": [
        {"start": 0.0, "end": 2.5, "text": "Rates stay restrictive"},
        {"start": 2.5, "end": 4.0, "text": "until inflation cools."},
    ],
}


class FakeEngine:
    def __init__(self, text: str = RECORD["text"], language: str = "en") -> None:
        self.text = text
        self.language = language
        self.calls: list[str] = []

    def transcribe(self, audio, language=None):
        self.calls.append(str(audio))
        segments = RECORD["segments"] if self.text == RECORD["text"] else []
        return {"text": self.text, "language": self.language, "segments": segments}


class AsrFormattingTests(unittest.TestCase):
    def test_txt_and_json(self):
        self.assertEqual(asr.render(RECORD, "txt"), RECORD["text"] + "\n")
        payload = json.loads(asr.render(RECORD, "json"))
        self.assertEqual(payload["text"], RECORD["text"])
        self.assertEqual(len(payload["segments"]), 2)

    def test_srt_and_vtt_use_their_own_timestamp_decimal(self):
        srt = asr.render(RECORD, "srt")
        self.assertTrue(srt.startswith("1\n00:00:00,000 --> 00:00:02,500\n"))
        self.assertIn("2\n00:00:02,500 --> 00:00:04,000\n", srt)
        vtt = asr.render(RECORD, "vtt")
        self.assertTrue(vtt.startswith("WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.500\n"))

    def test_subtitles_require_timestamps(self):
        untimed = dict(RECORD, segments=[{"start": None, "end": None, "text": "hi"}])
        with self.assertRaises(ValueError):
            asr.render(untimed, "srt")
        # Plain text and JSON must still work when the backend cannot align.
        self.assertEqual(asr.render(untimed, "txt").strip(), RECORD["text"])
        self.assertIn('"text"', asr.render(untimed, "json"))

    def test_language_normalisation(self):
        self.assertIsNone(asr.normalize_language("auto"))
        self.assertIsNone(asr.normalize_language(""))
        self.assertEqual(asr.normalize_language("zh-CN"), "zh")
        self.assertEqual(asr.normalize_language("Mandarin"), "zh")
        self.assertEqual(asr.normalize_language("en-US"), "en")

    def test_backend_resolution_rejects_unavailable_mlx(self):
        with mock.patch.object(asr, "_has_mlx", return_value=False):
            self.assertEqual(asr.resolve_backend("auto"), "faster-whisper")
            with self.assertRaises(RuntimeError):
                asr.resolve_backend("mlx")
        with mock.patch.object(asr, "_has_mlx", return_value=True):
            self.assertEqual(asr.resolve_backend("auto"), "mlx")

    def test_faster_whisper_model_names_are_translated(self):
        self.assertEqual(
            asr.faster_whisper_name("mlx-community/whisper-large-v3-turbo-asr-fp16"),
            "large-v3-turbo",
        )
        self.assertEqual(asr.faster_whisper_name("small"), "small")

    def test_cli_writes_the_requested_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "cue.wav"
            audio.write_bytes(b"RIFF")
            out = Path(tmp) / "nested" / "cue.vtt"
            engine = FakeEngine()
            with mock.patch.object(asr, "Transcriber", return_value=engine), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout, \
                    contextlib.redirect_stderr(io.StringIO()):
                code = asr.main([
                    "--audio", str(audio),
                    "--out", str(out),
                    "--format", "vtt",
                    "--quiet",
                ])
            self.assertEqual(code, 0)
            self.assertTrue(out.read_text(encoding="utf-8").startswith("WEBVTT"))
            self.assertEqual(stdout.getvalue(), "")

    def test_missing_audio_fails_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                code = asr.main([
                    "--audio", str(Path(tmp) / "absent.wav"),
                    "--out", str(Path(tmp) / "out.txt"),
                ])
            self.assertEqual(code, 1)


class QaNormalisationTests(unittest.TestCase):
    def test_spelled_numbers_match_asr_digits(self):
        expected = qa._normalise("Seventy-five percent, or twenty twenty-six.", "en")
        actual = qa._normalise("75% or 2026", "en")
        self.assertEqual(expected, actual)

    def test_mandarin_numerals_match_arabic_digits(self):
        self.assertEqual(qa._normalise("二十五亿", "zh"), list("25亿"))
        self.assertEqual(qa._normalise("25亿", "zh"), list("25亿"))

    def test_edit_distance(self):
        self.assertEqual(qa._distance([], []), 0)
        self.assertEqual(qa._distance([], ["a"]), 1)
        self.assertEqual(qa._distance(["a", "b"], ["a", "c"]), 1)

    def test_repeat_flags(self):
        self.assertEqual(qa._repeat_flags(["the", "the", "market"], "en"), ["consecutive repeated word: the"])
        self.assertEqual(qa._repeat_flags(["it", "is", "it", "is", "late"], "en"), ["repeated phrase: it is"])
        self.assertEqual(qa._repeat_flags(["share", "prices", "share", "prices"], "en"), [])
        self.assertEqual(qa._repeat_flags(["one", "and", "one", "and", "two"], "en"), [])
        self.assertEqual(qa._repeat_flags(["market", "market", "market"], "en"), ["three repeated words: market"])
        # A repeat that is in the script is not a TTS stutter.
        self.assertEqual(
            qa._repeat_flags(["it", "is", "it", "is"], "en", ["it", "is", "it", "is"]),
            [],
        )


class QaReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.manifest = self.dir / "cues.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "language": "en",
                    "cues": [
                        {"id": "shot-1", "text": "Rates stay restrictive until inflation cools."},
                        {"id": "shot-2", "text": "Growth is slowing."},
                    ],
                }
            ),
            encoding="utf-8",
        )
        for cue in ("shot-1", "shot-2"):
            (self.dir / f"{cue}.wav").write_bytes(b"RIFF")
        self.args = mock.Mock(
            wer_threshold=0.30,
            cer_threshold=0.15,
            min_wpm=105,
            max_wpm=220,
            min_cps=2.0,
            max_cps=7.5,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_audio_paths_resolve_against_manifest_and_audio_dir(self):
        entry = {"id": "shot-1"}
        self.assertEqual(
            qa.resolve_audio(entry, 1, self.manifest, None),
            self.dir / "shot-1.wav",
        )
        self.assertEqual(
            qa.resolve_audio({"id": "shot-1", "audio": "sub/a.wav"}, 1, self.manifest, None),
            self.dir / "sub" / "a.wav",
        )
        self.assertEqual(
            qa.resolve_audio({"id": "shot-1"}, 1, self.manifest, self.dir / "other"),
            self.dir / "other" / "shot-1.wav",
        )

    def test_matching_cue_passes(self):
        with mock.patch.object(qa, "_duration", return_value=4.0), \
                mock.patch.object(qa, "_levels", return_value=(-18.0, -3.0)):
            result = qa.check_cue(
                {"id": "shot-1", "text": "Rates stay restrictive until inflation cools."},
                self.dir / "shot-1.wav",
                FakeEngine(),
                "en",
                self.args,
            )
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["wer"], 0.0)

    def test_mismatch_and_short_audio_fail(self):
        with mock.patch.object(qa, "_duration", return_value=0.4), \
                mock.patch.object(qa, "_levels", return_value=(-18.0, -3.0)):
            result = qa.check_cue(
                {"id": "shot-2", "text": "Growth is slowing."},
                self.dir / "shot-2.wav",
                FakeEngine("completely unrelated words"),
                "en",
                self.args,
            )
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("too short" in error for error in result["errors"]))
        self.assertTrue(any("ASR mismatch" in error for error in result["errors"]))

    def test_missing_audio_is_reported_without_calling_asr(self):
        engine = FakeEngine()
        result = qa.check_cue(
            {"id": "shot-9", "text": "Anything."},
            self.dir / "shot-9.wav",
            engine,
            "en",
            self.args,
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(engine.calls, [])

    def test_cli_reuses_one_engine_and_fails_on_problems(self):
        out = self.dir / "qa.json"
        engine = FakeEngine()
        with mock.patch.object(qa, "Transcriber", return_value=engine), \
                mock.patch.object(qa, "_duration", return_value=4.0), \
                mock.patch.object(qa, "_levels", return_value=(-18.0, -3.0)), \
                contextlib.redirect_stdout(io.StringIO()):
            code = qa.main(["--manifest", str(self.manifest), "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 1, "cue 2 does not match the fake transcript")
        self.assertEqual(report["checked"], 2)
        self.assertGreaterEqual(report["problems"], 1)
        self.assertEqual(len(engine.calls), 2)

    def test_cli_passes_every_cue(self):
        out = self.dir / "qa.json"
        manifest = self.dir / "one.json"
        manifest.write_text(
            json.dumps({"language": "en", "cues": [{"id": "shot-1", "text": "Rates stay restrictive until inflation cools."}]}),
            encoding="utf-8",
        )
        with mock.patch.object(qa, "Transcriber", return_value=FakeEngine()), \
                mock.patch.object(qa, "_duration", return_value=4.0), \
                mock.patch.object(qa, "_levels", return_value=(-18.0, -3.0)), \
                contextlib.redirect_stdout(io.StringIO()):
            code = qa.main(["--manifest", str(manifest), "--out", str(out)])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
