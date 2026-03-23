import argparse
import csv
import json
import logging
import tarfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import requests
from datasets import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from scripts.asr_eval_utils import compute_wer, set_seed, write_csv, write_jsonl


LOGGER = logging.getLogger(__name__)


def load_wav_mono_float32(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Whisper model on Hindi ASR datasets.")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-small")
    parser.add_argument("--processed_jsonl", type=str, default="project/data/real_run_full_v2/metadata/dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="project/results/eval_baseline")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_subset", type=int, default=200)
    parser.add_argument("--max_fleurs_test", type=int, default=-1)
    parser.add_argument("--language", type=str, default="hindi")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--dedupe_consecutive_words", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def dedupe_consecutive_words(text: str) -> str:
    words = str(text or "").strip().split()
    if not words:
        return ""

    out = [words[0]]
    for word in words[1:]:
        if word != out[-1]:
            out.append(word)
    return " ".join(out)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_processed_dataset(jsonl_path: Path, max_samples: int) -> Dataset:
    ds = Dataset.from_json(str(jsonl_path))
    if max_samples > 0:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def load_fleurs_hi_test(max_samples: int) -> Dataset:
    base_dir = Path("project/data/fleurs_hi_test")
    audio_dir = base_dir / "audio"
    tsv_path = base_dir / "test.tsv"
    tar_path = base_dir / "test.tar.gz"

    base_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    tsv_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hi_in/test.tsv"
    tar_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hi_in/audio/test.tar.gz"

    if not tsv_path.exists():
        LOGGER.info("Downloading FLEURS Hindi test TSV...")
        resp = requests.get(tsv_url, timeout=120)
        resp.raise_for_status()
        tsv_path.write_bytes(resp.content)

    if not any(audio_dir.glob("*.wav")):
        if not tar_path.exists():
            LOGGER.info("Downloading FLEURS Hindi test audio tarball...")
            with requests.get(tar_url, timeout=120, stream=True) as resp:
                resp.raise_for_status()
                with tar_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

        LOGGER.info("Extracting FLEURS Hindi test audio...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=audio_dir)

    rows: List[Dict] = []
    test_audio_dir = audio_dir / "test"

    # Columns observed in FLEURS TSV (no header):
    # id, audio_filename, raw_transcription, transcription, words, num_samples, gender
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) < 4:
                continue
            sample_id = parts[0].strip()
            audio_name = parts[1].strip()
            transcription = parts[3].strip()
            audio_path = test_audio_dir / audio_name
            if not audio_path.exists():
                continue
            rows.append({"id": sample_id, "audio": str(audio_path), "transcription": transcription})

    if max_samples > 0:
        rows = rows[:max_samples]

    ds = Dataset.from_list(rows)
    return ds


def transcribe_dataset(
    dataset: Dataset,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    batch_size: int,
    language: str,
    task: str,
    text_key: str,
    id_key: str,
    dedupe_consecutive_words_enabled: bool,
) -> List[Dict]:
    records: List[Dict] = []

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        arrays = []
        for row in batch:
            audio, sample_rate = load_wav_mono_float32(row["audio"])
            if sample_rate != 16000:
                resampler = torch.nn.functional.interpolate(
                    torch.tensor(audio, dtype=torch.float32).view(1, 1, -1),
                    size=int(round(len(audio) * 16000 / sample_rate)),
                    mode="linear",
                    align_corners=False,
                )
                audio = resampler.view(-1).numpy()
            arrays.append(audio)

        inputs = processor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_features = inputs.input_features.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for row, pred in zip(batch, preds):
            rec_id = str(row[id_key]) if id_key in row else str(start)
            reference = str(row[text_key])
            prediction = pred.strip()
            if dedupe_consecutive_words_enabled:
                prediction = dedupe_consecutive_words(prediction)
            records.append(
                {
                    "id": rec_id,
                    "reference": reference,
                    "prediction": prediction,
                }
            )

    return records


def evaluate_split(name: str, records: List[Dict]) -> Dict:
    refs = [r["reference"] for r in records]
    preds = [r["prediction"] for r in records]
    wer, breakdown = compute_wer(refs, preds)
    return {
        "split": name,
        "samples": len(records),
        "wer": round(wer, 6),
        "substitutions": breakdown.substitutions,
        "deletions": breakdown.deletions,
        "insertions": breakdown.insertions,
        "ref_words": breakdown.ref_words,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    processed_ds = load_processed_dataset(Path(args.processed_jsonl), max_samples=args.max_train_subset)
    fleurs_ds = load_fleurs_hi_test(max_samples=args.max_fleurs_test)

    LOGGER.info("Processed subset samples: %d", len(processed_ds))
    LOGGER.info("FLEURS-hi test samples: %d", len(fleurs_ds))

    processed_records = transcribe_dataset(
        dataset=processed_ds,
        processor=processor,
        model=model,
        batch_size=args.batch_size,
        language=args.language,
        task=args.task,
        text_key="text",
        id_key="id",
        dedupe_consecutive_words_enabled=args.dedupe_consecutive_words,
    )

    fleurs_records = transcribe_dataset(
        dataset=fleurs_ds,
        processor=processor,
        model=model,
        batch_size=args.batch_size,
        language=args.language,
        task=args.task,
        text_key="transcription",
        id_key="id",
        dedupe_consecutive_words_enabled=args.dedupe_consecutive_words,
    )

    processed_metrics = evaluate_split("processed_subset", processed_records)
    fleurs_metrics = evaluate_split("fleurs_hi_test", fleurs_records)

    write_jsonl(processed_records, output_dir / "predictions_processed_subset.jsonl")
    write_jsonl(fleurs_records, output_dir / "predictions_fleurs_hi_test.jsonl")

    summary_rows = [
        {
            "model": args.model_name_or_path,
            "dataset": "processed_subset",
            "samples": processed_metrics["samples"],
            "wer": processed_metrics["wer"],
        },
        {
            "model": args.model_name_or_path,
            "dataset": "FLEURS-hi-test",
            "samples": fleurs_metrics["samples"],
            "wer": fleurs_metrics["wer"],
        },
    ]
    write_csv(summary_rows, output_dir / "wer_summary.csv")

    full_summary = {
        "model": args.model_name_or_path,
        "device": device,
        "dedupe_consecutive_words": bool(args.dedupe_consecutive_words),
        "processed_subset": processed_metrics,
        "fleurs_hi_test": fleurs_metrics,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(full_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
