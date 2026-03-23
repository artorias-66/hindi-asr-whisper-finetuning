# Hindi ASR Assignment (Josh Talks)

This repository contains an end-to-end Hindi ASR assignment submission covering preprocessing, baseline evaluation, fine-tuning, error analysis, targeted post-processing fixes, text cleanup, spelling-quality triage, and lattice-based evaluation.

## What Is Included

- End-to-end pipeline scripts for Q1-Q4
- Final assignment report in Markdown and PDF
- Reproducible outputs under the `results/` folder
- Preprocessing metadata and cleaned manifest artifacts under `data/real_run_full_v2/metadata/`

## Key Results

- FLEURS-hi WER:
  - Baseline (`openai/whisper-small`): 0.582006
  - Fine-tuned: 0.467227
  - Fine-tuned + dedup fix: 0.451988
- Processed subset WER:
  - Baseline: 1.557269
  - Fine-tuned: 0.821271
  - Fine-tuned + dedup fix: 0.609503

## Project Structure

- `preprocess.py`: Q1(a) preprocessing entrypoint
- `evaluate.py`: Q1(b,c,g) evaluation + WER + optional dedup fix
- `train.py`: Q1 fine-tuning script
- `analyze_errors.py`: Q1(d,e,f,g) error sampling/taxonomy/fix analysis
- `q2_generate_raw_asr.py`: Q2 raw ASR generation
- `q2_cleanup_pipeline.py`: Q2 number normalization + English tagging
- `q3_word_quality.py`: Q3 spelling-quality classification
- `q4_from_task_csv.py`: Q4 lattice-based evaluation from provided task CSV
- `results/assignment_answers.md`: Final narrative submission
- `results/assignment_answers.pdf`: PDF version of submission report

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Reproduce Core Steps

### Q1(a): Preprocess data

```powershell
python preprocess.py \
  --manifest "data/FT Data - data.csv" \
  --output_dir data/real_run_full_v2 \
  --audio_url_col rec_url_gcp \
  --transcription_url_col transcription_url_gcp \
  --duration_col duration \
  --duration_unit s \
  --speaker_col user_id \
  --id_col recording_id \
  --request_timeout_sec 20
```

### Q1(b,c): Baseline evaluation

```powershell
python evaluate.py \
  --model_name_or_path openai/whisper-small \
  --processed_jsonl "data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_dir results/eval_baseline_full \
  --max_train_subset 200 \
  --max_fleurs_test -1 \
  --batch_size 8
```

### Q1: Fine-tune and evaluate

```powershell
python train.py

python evaluate.py \
  --model_name_or_path "models/whisper-small-hi-ft" \
  --processed_jsonl "data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_dir results/eval_finetuned_v2 \
  --max_train_subset 200 \
  --max_fleurs_test -1 \
  --batch_size 8
```

### Q1(g): Evaluate with dedup fix

```powershell
python evaluate.py \
  --model_name_or_path "models/whisper-small-hi-ft" \
  --processed_jsonl "data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_dir results/eval_finetuned_v2_dedup \
  --max_train_subset 200 \
  --max_fleurs_test -1 \
  --batch_size 8 \
  --dedupe_consecutive_words
```

### Q2

```powershell
python q2_generate_raw_asr.py \
  --model_name_or_path openai/whisper-small \
  --processed_jsonl "data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_jsonl results/q2/raw_asr_pretrained.jsonl \
  --max_samples -1 \
  --batch_size 8

python q2_cleanup_pipeline.py
```

### Q3

```powershell
python q3_word_quality.py \
  --word_list_csv "data/Unique Words Data - Sheet1.csv" \
  --word_column word \
  --output_dir results/q3_external
```

### Q4

```powershell
python q4_from_task_csv.py \
  --input_csv "data/Question 4 - Task.csv" \
  --output_dir results/q4_task
```

## Final Deliverables

- Main report: `results/assignment_answers.md`
- PDF report: `results/assignment_answers.pdf`
- Submission index: `results/q1_submission_index.md`
- Q2 outputs: `results/q2/`
- Q3 outputs: `results/q3_external/`
- Q4 outputs: `results/q4_task/`

## Notes

- Large model and processed audio artifacts are ignored in git to keep repository size practical.
- If GPU is available, PyTorch CUDA builds are recommended for evaluation and training speed.
