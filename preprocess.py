import argparse
import json
import logging
import random
from pathlib import Path

from scripts.preprocess_utils import ProcessConfig, run_preprocessing


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Hindi ASR records into Whisper-ready HF dataset format."
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file (csv/json/jsonl/parquet).")
    parser.add_argument("--output_dir", type=str, default="project/data", help="Output directory for data artifacts.")
    parser.add_argument("--audio_url_col", type=str, default="rec_url_gcp", help="Audio URL column name in manifest.")
    parser.add_argument(
        "--transcription_url_col",
        type=str,
        default="transcription_url",
        help="Transcription JSON URL column name in manifest.",
    )
    parser.add_argument("--start_col", type=str, default="start", help="Start timestamp column name.")
    parser.add_argument("--end_col", type=str, default="end", help="End timestamp column name.")
    parser.add_argument("--duration_col", type=str, default="duration", help="Duration column name used when start/end are unavailable.")
    parser.add_argument(
        "--duration_unit",
        type=str,
        default="auto",
        choices=["auto", "s", "ms", "cs"],
        help="Duration unit for duration_col. Use cs for centiseconds (e.g., 443 -> 4.43s).",
    )
    parser.add_argument("--speaker_col", type=str, default="speaker_id", help="Speaker id column name.")
    parser.add_argument("--id_col", type=str, default=None, help="Optional sample id column name.")
    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg", help="Path or command name for ffmpeg.")
    parser.add_argument("--request_timeout_sec", type=int, default=30, help="HTTP request timeout in seconds.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for debugging small runs.")
    parser.add_argument("--merge_gap_sec", type=float, default=1.0, help="Merge adjacent segments if gap is below this threshold.")
    parser.add_argument("--min_duration_sec", type=float, default=1.0, help="Minimum allowed segment duration.")
    parser.add_argument("--max_duration_sec", type=float, default=20.0, help="Maximum allowed segment duration.")
    parser.add_argument("--min_text_chars", type=int, default=5, help="Minimum non-empty text length threshold.")
    parser.add_argument("--sanity_samples_to_print", type=int, default=5, help="How many random final samples to print for sanity checks.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    random.seed(args.seed)

    config = ProcessConfig(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        audio_url_col=args.audio_url_col,
        transcription_url_col=args.transcription_url_col,
        start_col=args.start_col,
        end_col=args.end_col,
        duration_col=args.duration_col,
        duration_unit=args.duration_unit,
        speaker_col=args.speaker_col,
        id_col=args.id_col,
        ffmpeg_bin=args.ffmpeg_bin,
        request_timeout_sec=args.request_timeout_sec,
        max_samples=args.max_samples,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        min_text_chars=args.min_text_chars,
        sanity_samples_to_print=args.sanity_samples_to_print,
        seed=args.seed,
    )

    result = run_preprocessing(config=config)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
