# Q1 Submission Index

## Final Narrative
- `project/results/assignment_answers.md`

## Preprocessing Outputs
- `project/data/real_run_full_v2/metadata/preprocess_summary.json`
- `project/data/real_run_full_v2/metadata/dataset.jsonl`
- `project/data/real_run_full_v2/metadata/skipped.jsonl`
- `project/data/real_run_full_v2/metadata/duration_distribution.csv`

## Baseline Evaluation (Pretrained Whisper-small)
- `project/results/eval_baseline_full/evaluation_summary.json`
- `project/results/eval_baseline_full/wer_summary.csv`
- `project/results/eval_baseline_full/predictions_processed_subset.jsonl`
- `project/results/eval_baseline_full/predictions_fleurs_hi_test.jsonl`

## Fine-tuned Evaluation (Final Model)
- `project/results/eval_finetuned_v2/evaluation_summary.json`
- `project/results/eval_finetuned_v2/wer_summary.csv`
- `project/results/eval_finetuned_v2/predictions_processed_subset.jsonl`
- `project/results/eval_finetuned_v2/predictions_fleurs_hi_test.jsonl`

## Integrated Fix Evaluation (Dedup Enabled)
- `project/results/eval_finetuned_v2_dedup/evaluation_summary.json`
- `project/results/eval_finetuned_v2_dedup/wer_summary.csv`
- `project/results/eval_finetuned_v2_dedup/predictions_processed_subset.jsonl`
- `project/results/eval_finetuned_v2_dedup/predictions_fleurs_hi_test.jsonl`

## Error Analysis (Q1 d-g)
- `project/results/error_analysis_v2/analysis_summary.json`
- `project/results/error_analysis_v2/error_taxonomy_counts.csv`
- `project/results/error_analysis_v2/sample_30_errors.jsonl`
- `project/results/error_analysis_v2/sample_30_errors.csv`
- `project/results/error_analysis_v2/fix_repetition_target_subset.jsonl`

## Model Artifact
- `project/models/whisper-small-hi-ft/model.safetensors`
- `project/models/whisper-small-hi-ft/config.json`
- `project/models/whisper-small-hi-ft/tokenizer.json`
- `project/models/whisper-small-hi-ft/tokenizer_config.json`
- `project/models/whisper-small-hi-ft/processor_config.json`
- `project/models/whisper-small-hi-ft/generation_config.json`
- `project/models/whisper-small-hi-ft/training_args.bin`
- `project/models/whisper-small-hi-ft/train_metrics.json`

## Key Q1 Metrics Snapshot
- Baseline FLEURS-hi WER: `0.582006`
- Fine-tuned (final) FLEURS-hi WER: `0.467227`
- Fine-tuned + dedup FLEURS-hi WER: `0.451988`
- Baseline processed subset WER: `1.557269`
- Fine-tuned (final) processed subset WER: `0.821271`
- Fine-tuned + dedup processed subset WER: `0.609503`
