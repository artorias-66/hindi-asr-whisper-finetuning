import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw ASR outputs from pretrained Whisper for Q2.")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-small")
    parser.add_argument("--processed_jsonl", type=str, default="project/data/real_run_full_v2/metadata/dataset.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="project/results/q2/raw_asr_pretrained.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--language", type=str, default="hindi")
    parser.add_argument("--task", type=str, default="transcribe")
    return parser.parse_args()


def load_wav_mono_float32(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def load_dataset(path: Path, max_samples: int) -> Dataset:
    ds = Dataset.from_json(str(path))
    if max_samples > 0:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    ds = load_dataset(Path(args.processed_jsonl), args.max_samples)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

    out_rows: List[Dict] = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds.select(range(start, min(start + args.batch_size, len(ds))))

        arrays: List[np.ndarray] = []
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

        inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for row, pred in zip(batch, preds):
            out_rows.append(
                {
                    "id": str(row["id"]),
                    "audio": str(row["audio"]),
                    "reference": str(row["text"]),
                    "raw_prediction": pred.strip(),
                }
            )

    write_jsonl(out_rows, Path(args.output_jsonl))
    print(json.dumps({"samples": len(out_rows), "output": args.output_jsonl, "device": device}, ensure_ascii=False))


if __name__ == "__main__":
    main()
