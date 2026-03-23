import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


LOGGER = logging.getLogger(__name__)


def load_wav_mono_float32(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on cleaned Hindi dataset.")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-small")
    parser.add_argument("--processed_jsonl", type=str, default="project/data/real_run_full_v2/metadata/dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="project/models/whisper-small-hi-ft")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--eval_steps", type=int, default=55)
    parser.add_argument("--save_steps", type=int, default=55)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--generation_max_length", type=int, default=225)
    parser.add_argument("--language", type=str, default="hindi")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_split_dataset(args: argparse.Namespace) -> DatasetDict:
    ds = Dataset.from_json(args.processed_jsonl)
    if args.max_train_samples > 0:
        ds = ds.select(range(min(len(ds), args.max_train_samples)))

    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(len(eval_ds), args.max_eval_samples)))

    return DatasetDict({"train": train_ds, "eval": eval_ds})


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    datasets = load_split_dataset(args)

    def prepare_batch(batch: Dict) -> Dict:
        audio, sample_rate = load_wav_mono_float32(batch["audio"])
        if sample_rate != 16000:
            resampler = torch.nn.functional.interpolate(
                torch.tensor(audio, dtype=torch.float32).view(1, 1, -1),
                size=int(round(len(audio) * 16000 / sample_rate)),
                mode="linear",
                align_corners=False,
            )
            audio = resampler.view(-1).numpy()

        batch["input_features"] = processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    vectorized = datasets.map(
        prepare_batch,
        remove_columns=datasets["train"].column_names,
        num_proc=1,
        desc="Preparing train/eval features",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized["train"],
        eval_dataset=vectorized["eval"],
        data_collator=data_collator,
    )

    LOGGER.info("Train samples: %d | Eval samples: %d", len(vectorized["train"]), len(vectorized["eval"]))
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(train_result.metrics, f, indent=2)

    LOGGER.info("Training complete. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
