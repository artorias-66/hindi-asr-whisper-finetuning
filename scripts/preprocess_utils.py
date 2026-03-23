import hashlib
import importlib
import json
import logging
import math
import random
import re
import shutil
import subprocess
import unicodedata
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


TEXT_KEY_PRIORITY = [
    "text",
    "transcript",
    "transcription",
    "sentence",
    "utterance",
    "normalized_text",
    "display_text",
    "prediction",
    "raw_text",
]

FILLER_TEXTS = {
    "हूं",
    "हूँ",
    "जी",
    "हाँ",
    "हां",
    "अच्छा",
}


@dataclass
class ProcessConfig:
    manifest_path: Path
    output_dir: Path
    audio_url_col: str = "rec_url_gcp"
    transcription_url_col: str = "transcription_url"
    start_col: str = "start"
    end_col: str = "end"
    duration_col: str = "duration"
    duration_unit: str = "auto"
    speaker_col: str = "speaker_id"
    id_col: Optional[str] = None
    ffmpeg_bin: str = "ffmpeg"
    request_timeout_sec: int = 30
    max_samples: Optional[int] = None
    merge_gap_sec: float = 1.0
    min_duration_sec: float = 1.0
    max_duration_sec: float = 20.0
    min_text_chars: int = 5
    sanity_samples_to_print: int = 5
    seed: int = 13


@dataclass
class RecordResult:
    sample_id: str
    audio_path: Optional[str]
    text: Optional[str]
    duration_sec: Optional[float]
    skipped_reason: Optional[str]


@dataclass
class SegmentRecord:
    order: int
    id: str
    audio: str
    text: str
    start: float
    end: float
    duration: float
    speaker_id: str
    audio_url: str


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "raw_audio": output_dir / "raw" / "audio",
        "raw_transcriptions": output_dir / "raw" / "transcriptions",
        "processed_audio": output_dir / "processed" / "wavs",
        "metadata": output_dir / "metadata",
        "hf_dataset": output_dir / "hf_dataset",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(manifest_path)
    elif suffix in {".jsonl", ".jl"}:
        df = pd.read_json(manifest_path, lines=True)
    elif suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
        if isinstance(content, list):
            df = pd.DataFrame(content)
        elif isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
            df = pd.DataFrame(content["data"])
        else:
            raise ValueError("JSON manifest must be a list of records or contain a 'data' list.")
    elif suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
    else:
        raise ValueError(f"Unsupported manifest format: {suffix}")

    if df.empty:
        raise ValueError("Manifest is empty.")

    return df


def make_sample_id(row: pd.Series, id_col: Optional[str], idx: int) -> str:
    if id_col and id_col in row and pd.notna(row[id_col]):
        return str(row[id_col])

    stable_fields = [str(row.get("rec_url_gcp", "")), str(row.get("transcription_url", "")), str(idx)]
    digest = hashlib.sha1("|".join(stable_fields).encode("utf-8")).hexdigest()[:16]
    return f"sample_{digest}"


def build_candidate_urls(url: str) -> List[str]:
    if not isinstance(url, str) or not url.strip():
        return []

    raw = url.strip()
    candidates = [raw]

    decoded = unquote(raw)
    if decoded != raw:
        candidates.append(decoded)

    if raw.startswith("gs://"):
        no_scheme = raw[len("gs://") :]
        bucket, _, obj = no_scheme.partition("/")
        if bucket and obj:
            candidates.append(f"https://storage.googleapis.com/{bucket}/{obj}")
            candidates.append(f"https://storage.cloud.google.com/{bucket}/{obj}")

    if "storage.cloud.google.com" in raw:
        candidates.append(raw.replace("storage.cloud.google.com", "storage.googleapis.com"))

    if "storage.googleapis.com" in raw and "?alt=media" not in raw:
        sep = "&" if "?" in raw else "?"
        candidates.append(f"{raw}{sep}alt=media")

    # Assignment-specific migration path: old joshtalks-data-collection URLs
    # can often be rewritten to upload_goai/{folder}/{filename}.
    try:
        parsed = urlparse(raw)
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2:
            folder = path_parts[-2]
            filename = path_parts[-1]
            if filename and folder:
                candidates.append(f"https://storage.googleapis.com/upload_goai/{folder}/{filename}")
    except Exception:  # noqa: BLE001
        pass

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _safe_extension_from_url(url: str, fallback: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower().strip()
    if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
        return suffix
    return fallback


def download_bytes(url: str, timeout_sec: int) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Hindi-ASR-Preprocess/1.0)"}
    response = requests.get(url, timeout=timeout_sec, headers=headers)
    response.raise_for_status()
    return response.content


def download_with_fallback_urls(url: str, output_path: Path, timeout_sec: int) -> Tuple[bool, Optional[str], Optional[str]]:
    errors = []
    for candidate in build_candidate_urls(url):
        try:
            payload = download_bytes(candidate, timeout_sec=timeout_sec)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(payload)
            return True, candidate, None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate} -> {exc}")

    if not errors:
        return False, None, "empty-or-invalid-url"
    return False, None, " | ".join(errors)


def load_json_with_fallback_urls(url: str, local_path: Path, timeout_sec: int) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    ok, resolved_url, err = download_with_fallback_urls(url=url, output_path=local_path, timeout_sec=timeout_sec)
    if not ok:
        return None, None, err

    try:
        with local_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, (dict, list)):
            return data, resolved_url, None
        return None, resolved_url, f"unsupported-transcription-json-type: {type(data).__name__}"
    except Exception as exc:  # noqa: BLE001
        return None, resolved_url, f"json-parse-error: {exc}"


def _extract_first_text(node: Any) -> Optional[str]:
    if isinstance(node, str):
        stripped = node.strip()
        return stripped if stripped else None

    if isinstance(node, dict):
        for key in TEXT_KEY_PRIORITY:
            if key in node:
                value = _extract_first_text(node[key])
                if value:
                    return value

        for _, value in node.items():
            nested = _extract_first_text(value)
            if nested:
                return nested
        return None

    if isinstance(node, list):
        for item in node:
            nested = _extract_first_text(item)
            if nested:
                return nested
    return None


def normalize_text(text: str) -> str:
    # Keep Devanagari and lexical content intact while removing control artifacts.
    text = text.replace("\ufeff", " ")

    cleaned_chars: List[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category in {"Cc", "Cf"} and char not in {"\t", "\n", "\r"}:
            continue
        cleaned_chars.append(char)

    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"[\t\r\n]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def duration_to_seconds(duration_value: float, duration_unit: str) -> float:
    unit = (duration_unit or "auto").strip().lower()
    if unit == "s":
        return duration_value
    if unit == "ms":
        return duration_value / 1000.0
    if unit == "cs":
        return duration_value / 100.0

    # Auto mode heuristic: values above 60 are almost always milliseconds in this dataset.
    if duration_value > 60.0:
        return duration_value / 1000.0
    return duration_value


def has_valid_duration(start: float, end: float, min_sec: float, max_sec: float) -> Tuple[bool, float]:
    duration = end - start
    return min_sec <= duration <= max_sec, duration


def is_low_content_text(text: str, min_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    return stripped in FILLER_TEXTS


def extract_transcript_segments(
    transcript_json: Any,
    fallback_start: Optional[float],
    fallback_end: Optional[float],
    fallback_speaker: str,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    nodes: List[Any]
    if isinstance(transcript_json, list):
        nodes = transcript_json
    elif isinstance(transcript_json, dict) and isinstance(transcript_json.get("data"), list):
        nodes = transcript_json["data"]
    else:
        nodes = [transcript_json]

    for item in nodes:
        if isinstance(item, dict):
            text = _extract_first_text(item)
            if not text:
                continue
            start = parse_float(item.get("start"))
            end = parse_float(item.get("end"))
            speaker = str(item.get("speaker_id", fallback_speaker) or fallback_speaker).strip() or fallback_speaker
            segments.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker_id": speaker,
                }
            )
        else:
            text = _extract_first_text(item)
            if text:
                segments.append(
                    {
                        "text": text,
                        "start": None,
                        "end": None,
                        "speaker_id": fallback_speaker,
                    }
                )

    if not segments:
        fallback_text = _extract_first_text(transcript_json)
        if fallback_text:
            segments = [
                {
                    "text": fallback_text,
                    "start": fallback_start,
                    "end": fallback_end,
                    "speaker_id": fallback_speaker,
                }
            ]

    for segment in segments:
        if segment["start"] is None or segment["end"] is None:
            segment["start"] = fallback_start
            segment["end"] = fallback_end
        if not segment.get("speaker_id"):
            segment["speaker_id"] = fallback_speaker

    cleaned = [seg for seg in segments if seg.get("start") is not None and seg.get("end") is not None and seg["end"] > seg["start"]]
    return cleaned


def concat_wav_files(
    input_wavs: List[Path],
    output_wav: Path,
    gaps_sec: Optional[List[float]] = None,
) -> Tuple[bool, Optional[str]]:
    if not input_wavs:
        return False, "no-input-wavs"

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    gaps_sec = gaps_sec or []

    try:
        with wave.open(str(input_wavs[0]), "rb") as first:
            nchannels = first.getnchannels()
            sampwidth = first.getsampwidth()
            framerate = first.getframerate()
            comptype = first.getcomptype()
            compname = first.getcompname()

        with wave.open(str(output_wav), "wb") as out:
            out.setnchannels(nchannels)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.setcomptype(comptype, compname)

            for idx, wav_path in enumerate(input_wavs):
                with wave.open(str(wav_path), "rb") as in_wav:
                    if (
                        in_wav.getnchannels() != nchannels
                        or in_wav.getsampwidth() != sampwidth
                        or in_wav.getframerate() != framerate
                    ):
                        return False, f"incompatible-wav-params: {wav_path}"
                    out.writeframes(in_wav.readframes(in_wav.getnframes()))

                if idx < len(gaps_sec):
                    gap = max(0.0, float(gaps_sec[idx]))
                    if gap > 0:
                        silence_frames = int(round(gap * framerate))
                        silence_bytes = b"\x00" * silence_frames * nchannels * sampwidth
                        out.writeframes(silence_bytes)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def merge_adjacent_segments(
    segments: List[SegmentRecord],
    merge_gap_sec: float,
    output_audio_dir: Path,
) -> Tuple[List[SegmentRecord], List[Dict[str, Any]]]:
    if not segments:
        return [], []

    ordered = sorted(segments, key=lambda rec: rec.order)

    merged: List[SegmentRecord] = []
    merge_logs: List[Dict[str, Any]] = []

    current_group: List[SegmentRecord] = [ordered[0]]
    for segment in ordered[1:]:
        last = current_group[-1]
        gap = segment.start - last.end
        can_merge = (
            segment.speaker_id == last.speaker_id
            and 0.0 <= gap < merge_gap_sec
        )
        if can_merge:
            current_group.append(segment)
        else:
            merged_segment, log_item = _finalize_segment_group(current_group, output_audio_dir)
            merged.append(merged_segment)
            if log_item:
                merge_logs.append(log_item)
            current_group = [segment]

    merged_segment, log_item = _finalize_segment_group(current_group, output_audio_dir)
    merged.append(merged_segment)
    if log_item:
        merge_logs.append(log_item)

    return merged, merge_logs


def _finalize_segment_group(group: List[SegmentRecord], output_audio_dir: Path) -> Tuple[SegmentRecord, Optional[Dict[str, Any]]]:
    if len(group) == 1:
        return group[0], None

    first = group[0]
    last = group[-1]
    merged_id = f"merged_{first.id}_{last.id}"
    merged_wav = output_audio_dir / f"{merged_id}.wav"

    gaps = []
    for prev, nxt in zip(group[:-1], group[1:]):
        gaps.append(max(0.0, nxt.start - prev.end))

    ok, err = concat_wav_files(
        input_wavs=[Path(item.audio) for item in group],
        output_wav=merged_wav,
        gaps_sec=gaps,
    )
    if not ok:
        LOGGER.warning("Failed to concatenate wavs for %s; fallback to first segment audio. Error=%s", merged_id, err)
        merged_audio = first.audio
    else:
        merged_audio = str(merged_wav.resolve())

    merged_record = SegmentRecord(
        order=first.order,
        id=merged_id,
        audio=merged_audio,
        text=normalize_text(" ".join(item.text for item in group)),
        start=first.start,
        end=last.end,
        duration=last.end - first.start,
        speaker_id=first.speaker_id,
        audio_url=first.audio_url,
    )
    return merged_record, {"merged_id": merged_id, "num_segments": len(group)}


def duration_counter_rows(durations: List[float], ndigits: int = 1) -> List[Dict[str, Any]]:
    rounded = [round(val, ndigits) for val in durations]
    counts = Counter(rounded)
    rows = [{"duration": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: x[0])]
    return rows


def top_words(texts: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in text.split():
            token = token.strip()
            if token:
                counter[token] += 1
    return [{"word": word, "count": count} for word, count in counter.most_common(top_k)]


def convert_to_wav_mono_16k(
    input_path: Path,
    output_path: Path,
    ffmpeg_bin: str,
    clip_start_sec: Optional[float] = None,
    clip_duration_sec: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_ffmpeg = ffmpeg_bin
    if shutil.which(ffmpeg_bin) is None:
        try:
            imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
            resolved_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            resolved_ffmpeg = ffmpeg_bin

    ffmpeg_cmd = [
        resolved_ffmpeg,
        "-y",
    ]
    if clip_start_sec is not None and clip_start_sec > 0:
        ffmpeg_cmd.extend(["-ss", f"{clip_start_sec:.3f}"])
    ffmpeg_cmd.extend(["-i", str(input_path)])
    if clip_duration_sec is not None and clip_duration_sec > 0:
        ffmpeg_cmd.extend(["-t", f"{clip_duration_sec:.3f}"])
    ffmpeg_cmd.extend(["-ac", "1", "-ar", "16000", "-vn", str(output_path)])
    try:
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return True, None
        ffmpeg_error = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
    except Exception as exc:  # noqa: BLE001
        ffmpeg_error = f"ffmpeg execution error: {exc}"

    # Lightweight fallback: if input is already WAV, validate and copy/rewrite using stdlib.
    try:
        with wave.open(str(input_path), "rb") as in_wav:
            nchannels = in_wav.getnchannels()
            sampwidth = in_wav.getsampwidth()
            framerate = in_wav.getframerate()
            total_frames = in_wav.getnframes()

            start_frame = 0
            end_frame = total_frames
            if clip_start_sec is not None and clip_start_sec > 0:
                start_frame = max(0, int(round(clip_start_sec * framerate)))
            if clip_duration_sec is not None and clip_duration_sec > 0:
                end_frame = min(total_frames, start_frame + int(round(clip_duration_sec * framerate)))

            in_wav.setpos(min(start_frame, total_frames))
            frames = in_wav.readframes(max(0, end_frame - start_frame))

        if nchannels == 1 and framerate == 16000 and sampwidth == 2:
            with wave.open(str(output_path), "wb") as out_wav:
                out_wav.setnchannels(1)
                out_wav.setsampwidth(2)
                out_wav.setframerate(16000)
                out_wav.writeframes(frames)
            return True, None
    except Exception:  # noqa: BLE001
        pass

    try:
        torchaudio = importlib.import_module("torchaudio")

        waveform, sample_rate = torchaudio.load(str(input_path))
        if waveform.ndim != 2:
            return False, f"unexpected waveform shape: {tuple(waveform.shape)}"

        if clip_start_sec is not None and clip_start_sec > 0:
            start_idx = int(round(clip_start_sec * sample_rate))
        else:
            start_idx = 0
        if clip_duration_sec is not None and clip_duration_sec > 0:
            end_idx = start_idx + int(round(clip_duration_sec * sample_rate))
        else:
            end_idx = waveform.shape[1]
        start_idx = max(0, min(start_idx, waveform.shape[1]))
        end_idx = max(start_idx, min(end_idx, waveform.shape[1]))
        waveform = waveform[:, start_idx:end_idx]

        mono = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            mono = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(mono)

        torchaudio.save(str(output_path), mono, 16000, encoding="PCM_S", bits_per_sample=16)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{ffmpeg_error}; torchaudio fallback failed: {exc}"


def get_wav_duration_sec(wav_path: Path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        if rate <= 0:
            raise ValueError("invalid sampling rate in wav")
        return frames / float(rate)


def make_duration_bins(durations: List[float], bin_size_sec: float = 2.0) -> List[Dict[str, Any]]:
    if not durations:
        return []

    max_duration = max(durations)
    n_bins = max(1, int(math.ceil(max_duration / bin_size_sec)))
    bins: List[Dict[str, Any]] = []
    for idx in range(n_bins):
        start = idx * bin_size_sec
        end = (idx + 1) * bin_size_sec
        count = sum(1 for value in durations if start <= value < end)
        bins.append({"bin_start_sec": round(start, 3), "bin_end_sec": round(end, 3), "count": count})

    bins[-1]["count"] += sum(1 for value in durations if value >= bins[-1]["bin_end_sec"])
    return bins


def save_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_hf_dataset(records: List[Dict[str, Any]], output_path: Path) -> Tuple[bool, Optional[str]]:
    try:
        datasets_module = importlib.import_module("datasets")
        Audio = datasets_module.Audio
        Dataset = datasets_module.Dataset

        dataset = Dataset.from_list(records)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset.save_to_disk(str(output_path))
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def run_preprocessing(config: ProcessConfig) -> Dict[str, Any]:
    paths = ensure_dirs(config.output_dir)
    df = load_manifest(config.manifest_path)

    if config.audio_url_col not in df.columns:
        raise ValueError(f"Audio URL column missing: {config.audio_url_col}")
    if config.transcription_url_col not in df.columns:
        raise ValueError(f"Transcription URL column missing: {config.transcription_url_col}")

    has_start_end = config.start_col in df.columns and config.end_col in df.columns
    has_duration = config.duration_col in df.columns
    if not has_start_end and not has_duration:
        raise ValueError(
            f"Need either start/end columns ({config.start_col}, {config.end_col}) "
            f"or a duration column ({config.duration_col})."
        )
    if config.speaker_col not in df.columns:
        LOGGER.warning("Speaker column missing: %s. Falling back to 'unknown' speaker for all rows.", config.speaker_col)

    if config.max_samples is not None:
        df = df.head(config.max_samples)

    LOGGER.info("Loaded %d manifest rows", len(df))

    total_raw_samples = len(df)
    segment_records: List[SegmentRecord] = []
    skipped_records: List[Dict[str, Any]] = []
    skip_duration_count = 0
    skip_low_content_count = 0
    skip_clip_mismatch_count = 0
    raw_valid_durations: List[float] = []
    order_counter = 0

    iterator = tqdm(df.iterrows(), total=len(df), desc="Preprocessing segments")
    for idx, row in iterator:
        sample_id = make_sample_id(row=row, id_col=config.id_col, idx=int(idx))

        audio_url = str(row.get(config.audio_url_col, "") or "").strip()
        text_url = str(row.get(config.transcription_url_col, "") or "").strip()

        if not audio_url:
            skipped_records.append({"id": sample_id, "reason": "missing-audio-url"})
            continue
        if not text_url:
            skipped_records.append({"id": sample_id, "reason": "missing-transcription-url"})
            continue

        start = parse_float(row.get(config.start_col)) if config.start_col in df.columns else None
        end = parse_float(row.get(config.end_col)) if config.end_col in df.columns else None

        if (start is None or end is None) and has_duration:
            duration_val = parse_float(row.get(config.duration_col))
            if duration_val is not None and duration_val > 0:
                duration_sec = duration_to_seconds(duration_val, config.duration_unit)
                start = 0.0
                end = duration_sec

        speaker_id = str(row.get(config.speaker_col, "unknown") or "unknown").strip() or "unknown"

        fallback_start = start
        fallback_end = end

        audio_ext = _safe_extension_from_url(audio_url, fallback=".bin")
        raw_audio_path = paths["raw_audio"] / f"{sample_id}{audio_ext}"
        raw_transcription_path = paths["raw_transcriptions"] / f"{sample_id}.json"
        processed_wav_path = paths["processed_audio"] / f"{sample_id}.wav"

        ok_audio, resolved_audio_url, audio_err = download_with_fallback_urls(
            url=audio_url,
            output_path=raw_audio_path,
            timeout_sec=config.request_timeout_sec,
        )
        if not ok_audio:
            skipped_records.append(
                {
                    "id": sample_id,
                    "reason": "audio-download-failed",
                    "detail": audio_err,
                }
            )
            continue

        transcript_json, resolved_text_url, trans_err = load_json_with_fallback_urls(
            url=text_url,
            local_path=raw_transcription_path,
            timeout_sec=config.request_timeout_sec,
        )
        if transcript_json is None:
            skipped_records.append(
                {
                    "id": sample_id,
                    "reason": "transcription-download-or-parse-failed",
                    "detail": trans_err,
                }
            )
            continue

        transcript_segments = extract_transcript_segments(
            transcript_json=transcript_json,
            fallback_start=fallback_start,
            fallback_end=fallback_end,
            fallback_speaker=speaker_id,
        )
        if not transcript_segments:
            skipped_records.append(
                {
                    "id": sample_id,
                    "reason": "missing-transcription-segments",
                }
            )
            continue
        for seg_idx, seg in enumerate(transcript_segments):
            seg_start = parse_float(seg.get("start"))
            seg_end = parse_float(seg.get("end"))
            seg_speaker = str(seg.get("speaker_id", speaker_id) or speaker_id).strip() or speaker_id

            if seg_start is None or seg_end is None or seg_end <= seg_start:
                skipped_records.append({"id": f"{sample_id}_{seg_idx}", "reason": "invalid-timestamps"})
                continue

            raw_valid_durations.append(seg_end - seg_start)
            valid_duration, duration = has_valid_duration(
                start=seg_start,
                end=seg_end,
                min_sec=config.min_duration_sec,
                max_sec=config.max_duration_sec,
            )
            if not valid_duration:
                skip_duration_count += 1
                skipped_records.append(
                    {
                        "id": f"{sample_id}_{seg_idx}",
                        "reason": "duration-out-of-range",
                        "duration": round(duration, 4),
                    }
                )
                continue

            text = normalize_text(str(seg.get("text", "")))
            if not text:
                skipped_records.append({"id": f"{sample_id}_{seg_idx}", "reason": "empty-transcription-after-normalization"})
                continue

            if is_low_content_text(text=text, min_chars=config.min_text_chars):
                skip_low_content_count += 1
                skipped_records.append(
                    {
                        "id": f"{sample_id}_{seg_idx}",
                        "reason": "low-content-text",
                        "text": text,
                    }
                )
                continue

            segment_sample_id = f"{sample_id}_{seg_idx}"
            processed_segment_wav_path = paths["processed_audio"] / f"{segment_sample_id}.wav"
            ok_convert, convert_err = convert_to_wav_mono_16k(
                input_path=raw_audio_path,
                output_path=processed_segment_wav_path,
                ffmpeg_bin=config.ffmpeg_bin,
                clip_start_sec=float(seg_start),
                clip_duration_sec=float(duration),
            )
            if not ok_convert:
                skipped_records.append(
                    {
                        "id": segment_sample_id,
                        "reason": "audio-conversion-failed",
                        "detail": convert_err,
                    }
                )
                continue

            try:
                wav_duration = get_wav_duration_sec(processed_segment_wav_path)
            except Exception as exc:  # noqa: BLE001
                skipped_records.append(
                    {
                        "id": segment_sample_id,
                        "reason": "invalid-wav-after-conversion",
                        "detail": str(exc),
                    }
                )
                continue

            if abs(wav_duration - float(duration)) > 0.35:
                skip_clip_mismatch_count += 1
                skipped_records.append(
                    {
                        "id": segment_sample_id,
                        "reason": "clip-duration-mismatch",
                        "expected_duration": round(float(duration), 4),
                        "actual_duration": round(float(wav_duration), 4),
                    }
                )
                continue

            segment_records.append(
                SegmentRecord(
                    order=order_counter,
                    id=segment_sample_id,
                    audio=str(processed_segment_wav_path.resolve()),
                    text=text,
                    start=float(seg_start),
                    end=float(seg_end),
                    duration=round(duration, 4),
                    speaker_id=seg_speaker,
                    audio_url=resolved_audio_url or audio_url,
                )
            )
            order_counter += 1

    merged_records, merge_logs = merge_adjacent_segments(
        segments=segment_records,
        merge_gap_sec=config.merge_gap_sec,
        output_audio_dir=paths["processed_audio"],
    )

    final_records: List[Dict[str, Any]] = []
    final_durations: List[float] = []

    for merged in merged_records:
        valid_duration, duration = has_valid_duration(
            start=merged.start,
            end=merged.end,
            min_sec=config.min_duration_sec,
            max_sec=config.max_duration_sec,
        )
        if not valid_duration:
            skip_duration_count += 1
            skipped_records.append(
                {
                    "id": merged.id,
                    "reason": "duration-out-of-range-after-merge",
                    "duration": round(duration, 4),
                }
            )
            continue

        merged_text = normalize_text(merged.text)
        if not merged_text or is_low_content_text(merged_text, config.min_text_chars):
            skip_low_content_count += 1
            skipped_records.append(
                {
                    "id": merged.id,
                    "reason": "low-content-text-after-merge",
                    "text": merged_text,
                }
            )
            continue

        if not Path(merged.audio).exists():
            skipped_records.append(
                {
                    "id": merged.id,
                    "reason": "missing-audio-after-merge",
                    "audio": merged.audio,
                }
            )
            continue

        final_duration = round(duration, 4)
        final_records.append(
            {
                "id": merged.id,
                "audio": merged.audio,
                "text": merged_text,
                "duration": final_duration,
            }
        )
        final_durations.append(final_duration)

    metadata_path = paths["metadata"]
    valid_jsonl_path = metadata_path / "dataset.jsonl"
    skipped_jsonl_path = metadata_path / "skipped.jsonl"
    save_jsonl(final_records, valid_jsonl_path)
    save_jsonl(skipped_records, skipped_jsonl_path)
    save_jsonl(merge_logs, metadata_path / "merge_log.jsonl")

    hf_ok, hf_err = save_hf_dataset(
        records=[{"id": rec["id"], "audio": rec["audio"], "text": rec["text"]} for rec in final_records],
        output_path=paths["hf_dataset"],
    )

    duration_rows = duration_counter_rows(final_durations, ndigits=1)
    pd.DataFrame(duration_rows).to_csv(metadata_path / "duration_distribution.csv", index=False)

    before_hours = round(sum(raw_valid_durations) / 3600.0, 4) if raw_valid_durations else 0.0
    after_hours = round(sum(final_durations) / 3600.0, 4)
    removal_base_count = len(merged_records) if len(merged_records) > 0 else total_raw_samples
    removed_pct = round(100.0 * (1.0 - (len(final_records) / removal_base_count)), 2) if removal_base_count > 0 else 0.0
    top_10_words = top_words([rec["text"] for rec in final_records], top_k=10)

    stats = {
        "total_raw_samples": int(total_raw_samples),
        "after_merging": int(len(merged_records)),
        "final_samples": int(len(final_records)),
        "removed_percent": removed_pct,
        "skipped_duration": int(skip_duration_count),
        "skipped_low_content": int(skip_low_content_count),
        "skipped_clip_mismatch": int(skip_clip_mismatch_count),
        "total_skipped": int(len(skipped_records)),
        "total_hours_before_filtering": before_hours,
        "total_hours_after_filtering": after_hours,
        "total_hours": after_hours,
        "avg_duration": round(mean(final_durations), 4) if final_durations else 0.0,
        "avg_text_length_chars": round(mean([len(rec["text"]) for rec in final_records]), 2) if final_records else 0.0,
        "skipped_samples": int(len(skipped_records)),
        "hf_dataset_saved": hf_ok,
        "hf_dataset_error": hf_err,
        "top_10_words": top_10_words,
    }

    if final_durations:
        sorted_d = sorted(final_durations)
        p90_index = min(len(sorted_d) - 1, int(0.90 * len(sorted_d)))
        stats.update(
            {
                "duration_min_sec": round(min(final_durations), 4),
                "duration_max_sec": round(max(final_durations), 4),
                "duration_mean_sec": round(mean(final_durations), 4),
                "duration_median_sec": round(median(final_durations), 4),
                "duration_p90_sec": round(sorted_d[p90_index], 4),
            }
        )

    with (metadata_path / "preprocess_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    LOGGER.info("Raw samples: %d", stats["total_raw_samples"])
    LOGGER.info("After merging: %d", stats["after_merging"])
    LOGGER.info("Final samples: %d", stats["final_samples"])
    LOGGER.info("Removed data: %.2f%%", stats["removed_percent"])
    LOGGER.info("Total hours before filtering: %.4f", stats["total_hours_before_filtering"])
    LOGGER.info("Total hours after filtering: %.4f", stats["total_hours_after_filtering"])
    LOGGER.info("Skipped due to duration: %d", stats["skipped_duration"])
    LOGGER.info("Skipped due to low content: %d", stats["skipped_low_content"])
    LOGGER.info("Skipped due to clip mismatch: %d", stats["skipped_clip_mismatch"])
    if final_durations:
        LOGGER.info(
            "Duration stats (sec): min=%0.2f, median=%0.2f, mean=%0.2f, p90=%0.2f, max=%0.2f",
            stats["duration_min_sec"],
            stats["duration_median_sec"],
            stats["duration_mean_sec"],
            stats["duration_p90_sec"],
            stats["duration_max_sec"],
        )

    sample_count = min(config.sanity_samples_to_print, len(final_records))
    if sample_count > 0:
        rng = random.Random(config.seed)
        for rec in rng.sample(final_records, sample_count):
            LOGGER.info(
                "Sample | ID: %s | Duration: %.2f | Text length: %d chars | Text: %s",
                rec["id"],
                rec["duration"],
                len(rec["text"]),
                rec["text"],
            )

    if top_10_words:
        LOGGER.info("Top 10 most frequent words: %s", top_10_words)

    return {
        "stats": stats,
        "valid_records_path": str(valid_jsonl_path),
        "skipped_records_path": str(skipped_jsonl_path),
        "duration_distribution_path": str(metadata_path / "duration_distribution.csv"),
        "hf_dataset_path": str(paths["hf_dataset"]),
    }
