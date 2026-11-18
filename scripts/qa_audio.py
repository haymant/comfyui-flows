#!/usr/bin/env python3
"""Check synthesized speech against the script that produced it.

This is a gate, not a pronunciation oracle. It transcribes each cue with the
local ASR in ``scripts/asr.py`` and flags the failures that cost a re-render:
truncation, dropped or extra words, stutters, extreme pace, and bad levels.
Accent, timbre and naturalness still need a human.

The report is JSON so a pipeline can decide per cue what to re-synthesize.

Manifest
--------
::

    {
      "language": "en",
      "cues": [
        {"id": "shot-1", "text": "The expected narration.", "audio": "out/shot-1.wav"},
        {"id": "shot-2", "text": "…", "audio": "out/shot-2.wav"}
      ]
    }

``audio`` is resolved relative to the manifest. With ``--audio-dir`` and no
``audio`` field, ``<audio-dir>/<id>.wav`` is used. ``--audio-dir`` also serves as
the base for relative ``audio`` paths.

The ASR model is loaded once and reused for every cue: spawning a process per
cue costs several seconds each, which dominates a short-cue QA run.

Examples
--------
    python3 scripts/qa_audio.py --manifest out/cues.json --out out/qa.json
    python3 scripts/qa_audio.py --manifest out/cues.json --limit 3 -v
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from asr import Transcriber, DEFAULT_MLX_MODEL  # noqa: E402

ZH_PREFIXES = ("zh", "cmn", "mandarin", "chinese")

_ZH_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_ZH_UNITS = {"十": 10, "百": 100, "千": 1000}

_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_DECADES = {
    "twenties": "20s", "thirties": "30s", "forties": "40s", "fifties": "50s",
    "sixties": "60s", "seventies": "70s", "eighties": "80s", "nineties": "90s",
}


def _zh_int(value: str) -> int:
    total = 0
    number = 0
    for char in value:
        if char in _ZH_DIGITS:
            number = _ZH_DIGITS[char]
        else:
            total += (number or 1) * _ZH_UNITS[char]
            number = 0
    return total + number


def _zh_compact(value: str) -> str:
    """Keep magnitudes compact so both sides normalize to e.g. 25亿."""
    for unit in ("亿", "万"):
        if unit in value:
            left, right = value.split(unit, 1)
            head = _zh_compact(left) if left else "1"
            return head + unit + (_zh_compact(right) if right else "")
    return str(_zh_int(value))


def _en_number_words(text: str) -> str:
    """Turn spelled-out numbers into digits so ASR ("75%") and the script
    ("seventy-five percent") are not scored as a long substitution run."""
    text = text.replace("$", " ").replace(",", "")
    text = re.sub(r"[- ]point[- ]", " point ", text)
    text = re.sub(r"\ba\s+hundred\b", "one hundred", text)
    text = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine|\d+)\s+hundred\s+and\s+",
        r"\1 hundred ",
        text,
    )

    def century_decade(match: re.Match) -> str:
        prefix = "19" if match.group(1) == "nineteen" else "18"
        return prefix + _EN_DECADES[match.group(2)]

    text = re.sub(
        r"\b(nineteen|eighteen)[- ](" + "|".join(_EN_DECADES) + r")\b",
        century_decade,
        text,
    )
    for word, digits in _EN_DECADES.items():
        text = re.sub(rf"\b{word}\b", digits, text)

    def compound(match: re.Match) -> str:
        return str(_EN_TENS[match.group(1)] + _EN_UNITS[match.group(2)])

    text = re.sub(
        r"\b(" + "|".join(_EN_TENS) + r")[- ](" + "|".join(_EN_UNITS) + r")\b",
        compound,
        text,
    )

    def year(match: re.Match) -> str:
        prefix = 2000 if match.group(1) == "twenty" else 1900
        return str(prefix + int(match.group(2)))

    text = re.sub(r"\b(twenty|nineteen)[- ](\d{2})\b", year, text)

    def hundred(match: re.Match) -> str:
        return str(int(match.group(1)) * 100)

    text = re.sub(r"\b(\d+)\s+hundred\b", hundred, text)
    text = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)\s+hundred\b",
        lambda m: str(_EN_UNITS[m.group(1)] * 100),
        text,
    )
    for word, value in sorted(_EN_TENS.items(), key=lambda item: -len(item[0])):
        text = re.sub(rf"\b{word}\b", str(value), text)
    for word, value in sorted(_EN_UNITS.items(), key=lambda item: -len(item[0])):
        text = re.sub(rf"\b{word}\b", str(value), text)
    text = re.sub(r"(\d+)\s+point\s+(\d+)", r"\1.\2", text)

    def combine_hundreds(match: re.Match) -> str:
        head, tail = int(match.group(1)), int(match.group(2))
        return str(head + tail) if head % 100 == 0 else match.group(0)

    text = re.sub(r"\b(\d+)[- ]hundred\b", lambda m: str(int(m.group(1)) * 100)
                  if int(m.group(1)) < 10 else m.group(0), text)
    text = re.sub(r"\b(\d{3})\s+(\d{1,2})\b", combine_hundreds, text)
    for unit, factor in (("thousand", 1000), ("million", 1_000_000),
                         ("billion", 1_000_000_000), ("trillion", 1_000_000_000_000)):
        text = re.sub(
            rf"(\d+(?:\.\d+)?)[- ]{unit}\b",
            lambda m, f=factor: str(int(float(m.group(1)) * f)),
            text,
        )
    text = re.sub(r"\bper ?cent\b", " ", text)
    # Currency units add or drop a token depending on the ASR; the number is
    # what matters, so remove the unit words on both sides.
    text = re.sub(r"\b(dollars?|cents?|bucks?)\b", " ", text)
    return text


def _normalise(text: str, language: str) -> list[str]:
    text = text.lower()
    if language.lower().startswith(ZH_PREFIXES):
        # Whisper writes Arabic digits while the script may use Chinese
        # numerals. Keep the magnitude compact so 25亿 does not become
        # 2500000000.
        text = text.replace("百分之", "")
        text = re.sub(
            r"(?<!\d)[零〇一二两三四五六七八九十百千万亿]+",
            lambda match: _zh_compact(match.group(0)),
            text,
        )
        # Mixed-script names are transliterated inconsistently; ignore Latin
        # runs so a correct Chinese line is not failed by name spelling.
        text = re.sub(r"[A-Za-z]+", " ", text)
        return list(re.sub(r"[^\u3400-\u9fff0-9]", "", text))
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)?", _en_number_words(text))


def _distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (item != other))
            )
        previous = current
    return previous[-1]


def _duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except Exception:  # noqa: BLE001
        return 0.0


def _levels(path: Path) -> tuple[float, float]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        mean = re.search(r"mean_volume: ([-\d.]+)", result.stderr)
        peak = re.search(r"max_volume: ([-\d.]+)", result.stderr)
        return (
            float(mean.group(1)) if mean else -99.0,
            float(peak.group(1)) if peak else -99.0,
        )
    except Exception:  # noqa: BLE001
        return (-99.0, -99.0)


def _repeat_flags(tokens: list[str], language: str, expected: list[str] | None = None) -> list[str]:
    flags: list[str] = []
    zh = language.lower().startswith(ZH_PREFIXES)

    def contains_run(haystack: list[str], needle: list[str]) -> bool:
        size = len(needle)
        if not size or len(haystack) < size:
            return False
        return any(haystack[i:i + size] == needle for i in range(len(haystack) - size + 1))

    if not zh:
        # "earnings, earnings lift…" is grammatical; only a doubled function
        # word or a three-in-a-row stutter is a reliable TTS failure.
        stutter_words = {
            "the", "and", "a", "an", "to", "of", "in", "is", "it", "that",
            "this", "for", "on", "with", "as", "at", "but", "or", "if",
            "was", "were", "be", "been", "so", "we", "i",
        }
        for i in range(len(tokens) - 2):
            if tokens[i] == tokens[i + 1] == tokens[i + 2]:
                flags.append("three repeated words: " + tokens[i])
                break
        else:
            for i in range(len(tokens) - 1):
                if tokens[i] == tokens[i + 1] and tokens[i] in stutter_words:
                    flags.append("consecutive repeated word: " + tokens[i])
                    break
    else:
        # Chinese reduplication is normal; only three identical characters flag.
        for i in range(len(tokens) - 2):
            if tokens[i] == tokens[i + 1] == tokens[i + 2] and re.fullmatch(r"[\u3400-\u9fff]", tokens[i]):
                flags.append("three repeated characters: " + tokens[i])
                break
    # A repeated phrase. Three tokens need six to repeat twice; the two-token
    # case is only a stutter for function-word pairs, because "share prices,
    # share prices" is ordinary rhetorical repetition.
    stutter_pairs = {
        ("it", "is"), ("this", "is"), ("that", "is"), ("there", "is"),
        ("we", "have"), ("i", "have"), ("it", "was"), ("there", "was"),
        ("you", "can"), ("we", "are"), ("they", "are"), ("it", "will"),
    }
    for i in range(0, len(tokens) - 4 + 1):
        pair = (tokens[i], tokens[i + 1])
        if pair == (tokens[i + 2], tokens[i + 3]) and pair in stutter_pairs:
            if expected and contains_run(expected, list(pair) * 2):
                continue
            flags.append("repeated phrase: " + " ".join(pair))
            return flags
    size = 3
    for i in range(0, len(tokens) - size * 2 + 1):
        first = tokens[i:i + size]
        second = tokens[i + size:i + size * 2]
        if all(token.isdigit() for token in first):
            continue
        if first == second:
            # A repeat written in the script is not a TTS stutter: Mandarin
            # punctuation may be normalised away, fusing two sentences.
            if expected and contains_run(expected, first + second):
                continue
            flags.append("repeated phrase: " + " ".join(first))
            break
    return flags


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cues"), list):
        raise ValueError("manifest must be an object with a 'cues' array")
    return manifest


def resolve_audio(entry: dict, index: int, manifest_path: Path, audio_dir: Path | None) -> Path:
    cue_id = str(entry.get("id") or f"cue-{index}")
    audio = entry.get("audio")
    if audio:
        candidate = Path(audio)
        if not candidate.is_absolute():
            base = audio_dir or manifest_path.parent
            candidate = base / candidate
        return candidate
    base = audio_dir or manifest_path.parent
    return base / f"{cue_id}.wav"


def check_cue(
    entry: dict,
    audio: Path,
    engine: Transcriber,
    language: str,
    args: argparse.Namespace,
) -> dict:
    cue_id = str(entry.get("id") or audio.stem)
    expected_text = str(entry.get("text") or "").strip()
    if not expected_text:
        return {"id": cue_id, "status": "skip", "errors": [], "warnings": ["no text"]}
    if not audio.is_file():
        return {"id": cue_id, "status": "error", "errors": [f"missing audio: {audio}"], "warnings": []}

    record = engine.transcribe(audio, language=language)
    actual_text = str(record.get("text") or "").strip()
    expected = _normalise(expected_text, language)
    actual = _normalise(actual_text, language)
    distance = _distance(expected, actual)
    error_rate = distance / max(1, len(expected))
    duration = _duration(audio)
    is_zh = language.lower().startswith(ZH_PREFIXES)
    words = len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", expected_text))
    wpm = words / duration * 60 if duration > 0 else 0
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", expected_text)) if is_zh else 0
    cps = cjk_chars / duration if duration > 0 else 0
    mean_db, peak_db = _levels(audio)

    errors: list[str] = []
    warnings: list[str] = []
    if duration < 1.0:
        errors.append(f"audio too short ({duration:.2f}s)")
    if not actual:
        errors.append("ASR returned no speech")
    threshold = args.cer_threshold if is_zh else args.wer_threshold
    if error_rate > threshold:
        errors.append(f"ASR mismatch {error_rate:.1%}: {actual_text[:160]}")
    elif error_rate > threshold * 0.8:
        warnings.append(f"possible ASR/pronunciation mismatch {error_rate:.1%}")
    if duration:
        if is_zh and cjk_chars >= 12:
            if cps < args.min_cps or cps > args.max_cps:
                errors.append(
                    f"character pace outside range ({cps:.2f} chars/s, {duration:.2f}s)"
                )
        elif words >= 8 and not is_zh:
            upper = args.max_wpm if duration >= 8 else max(args.max_wpm, 240)
            if wpm < args.min_wpm or wpm > upper:
                errors.append(f"pace outside range ({wpm:.0f} wpm, {duration:.2f}s)")
    if not is_zh:
        # Keep number tokens separated: joining "5" and "2026" would create a
        # fake 52026 and report the year as missing.
        expected_numbers = re.findall(r"\d+(?:\.\d+)?", " ".join(expected))
        actual_numbers = re.findall(r"\d+(?:\.\d+)?", " ".join(actual))
        if expected_numbers:
            joined_actual = "".join(actual_numbers)
            missing = [
                n for n in expected_numbers
                if n not in actual_numbers and n not in joined_actual
            ]
            if missing:
                # WER is the authority; ASR renders numbers many ways
                # ("60/40", "6040", "sixty forty"), so this is a warning.
                warnings.append("number formatting differs: " + ", ".join(missing[:6]))
    repeat_flags = _repeat_flags(actual, language, expected)
    if repeat_flags:
        errors.extend(repeat_flags)
    if peak_db > -0.15:
        warnings.append(f"peak near full scale ({peak_db:.1f} dBFS)")
    if mean_db < -32:
        warnings.append(f"very quiet cue ({mean_db:.1f} dBFS)")
    if words >= 8 and actual and len(actual) < words * 0.55:
        errors.append(f"possible truncation: {len(actual)} ASR tokens vs {words} expected")

    return {
        "id": cue_id,
        "status": "fail" if errors else "pass",
        "language": record.get("language") or language,
        "wer": round(error_rate, 4),
        "duration": round(duration, 2),
        "wpm": round(wpm, 1),
        "cps": round(cps, 2),
        "mean_dbfs": mean_db,
        "peak_dbfs": peak_db,
        "expected": expected_text,
        "asr": actual_text,
        "errors": errors,
        "warnings": warnings,
        "audio": str(audio),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="cue manifest JSON")
    parser.add_argument("--out", required=True, help="report JSON path")
    parser.add_argument("--audio-dir", default=None, help="base directory for audio")
    parser.add_argument("--language", default=None, help="override the manifest language")
    parser.add_argument("--model", default=None, help=f"default MLX: {DEFAULT_MLX_MODEL}")
    parser.add_argument(
        "--backend", choices=["auto", "mlx", "faster-whisper"], default="auto"
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (faster-whisper)")
    parser.add_argument("--compute-type", default=None, help="float16 / int8 / …")
    parser.add_argument("--wer-threshold", type=float, default=0.30)
    parser.add_argument("--cer-threshold", type=float, default=0.15)
    parser.add_argument("--min-wpm", type=float, default=105)
    parser.add_argument("--max-wpm", type=float, default=220)
    parser.add_argument("--min-cps", type=float, default=2.0)
    parser.add_argument("--max-cps", type=float, default=7.5)
    parser.add_argument("--limit", type=int, default=0, help="check only the first N cues")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    try:
        manifest = load_manifest(manifest_path)
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        return 1

    language = args.language or str(manifest.get("language") or "en")
    entries = manifest["cues"]
    if args.limit:
        entries = entries[: args.limit]
    try:
        engine = Transcriber(
            model=args.model,
            backend=args.backend,
            device=args.device,
            compute_type=args.compute_type,
        )
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        return 1

    results = []
    problems = 0
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            results.append({"status": "error", "errors": ["cue is not an object"], "warnings": []})
            problems += 1
            continue
        audio = resolve_audio(entry, index, manifest_path, audio_dir)
        try:
            result = check_cue(entry, audio, engine, language, args)
        except Exception as error:  # noqa: BLE001
            result = {
                "id": str(entry.get("id") or f"cue-{index}"),
                "status": "error",
                "errors": [str(error)],
                "warnings": [],
            }
        if result["status"] not in {"pass", "skip"}:
            problems += 1
        results.append(result)
        if args.verbose:
            detail = ""
            if result.get("errors"):
                detail = " — " + "; ".join(result["errors"])
            print(
                f"[{index}/{len(entries)}] {result.get('id')}: {result['status']}"
                f" wer={result.get('wer')}{detail}",
                file=sys.stderr,
            )

    report = {
        "language": language,
        "manifest": str(manifest_path),
        "checked": len(results),
        "problems": problems,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"checked": len(results), "problems": problems, "out": str(out)},
            ensure_ascii=False,
        )
    )
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
