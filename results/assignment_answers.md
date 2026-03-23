# Task Assignment Submission Notes

## Candidate
- Name: (Fill before submission)
- Role: AI Researcher Intern - Speech & Audio
- Project: Josh Talks ASR Assignment
- Repository: https://github.com/artorias-66/hindi-asr-whisper-finetuning
- Last updated: 2026-03-23 (Q1-Q4 complete with provided external files)

## Executive Summary
- Completed end-to-end ASR workflow across preprocessing, baseline evaluation, fine-tuning, error analysis, text cleanup, word-quality triage, and lattice-based evaluation.
- Q1 final model quality improved over baseline:
  - FLEURS-hi WER: 0.582006 -> 0.467227 (fine-tuned) -> 0.451988 (fine-tuned + dedup fix)
  - processed subset WER: 1.557269 -> 0.821271 (fine-tuned) -> 0.609503 (fine-tuned + dedup fix)
- Q2 pipeline implemented and run on full 2390-sample raw ASR output:
  - number normalization and English-word tagging with saved before/after artifacts.
- Q3 completed using provided external file `Unique Words Data - Sheet1.csv`:
  - total unique words: 177421
  - correct spelling predicted: 170556
- Q4 completed using provided `Question 4 - Task.csv`:
  - lattice-based WER reduced unfair penalties for 5/6 model outputs and kept 1 unchanged.

## Q1
### (a) Data preprocessing and training readiness
#### What was implemented
- Robust URL download with fallback rewrite to assignment pattern:
  - `https://storage.googleapis.com/upload_goai/{folder}/{filename}`
- Audio processing to Whisper-ready format:
  - mono
  - 16 kHz
  - wav
- Segment-aware transcript handling:
  - transcription JSON is parsed as segment list
  - each segment uses its own `start`, `end`, `speaker_id`, `text`
- Segment clipping enforced during conversion:
  - audio clip is cut to segment timestamps
  - clip duration mismatch check is applied
- Quality filters:
  - duration keep range: 1.0s to 20.0s
  - low-content text filtering (short/filler)
- Merging:
  - adjacent segments merged for same speaker with gap < 1.0s
- Reproducibility and logs:
  - skipped samples written with explicit reason
  - summary JSON + duration distribution CSV + sanity samples

#### Real-data run command
```powershell
python project/preprocess.py \
  --manifest "project/data/FT Data - data.csv" \
  --output_dir project/data/real_run_full_v2 \
  --audio_url_col rec_url_gcp \
  --transcription_url_col transcription_url_gcp \
  --duration_col duration \
  --duration_unit s \
  --speaker_col user_id \
  --id_col recording_id \
  --request_timeout_sec 20
```

#### Final preprocessing metrics (real run)
- total_raw_samples: 104
- after_merging: 2985
- final_samples: 2390
- removed_percent: 19.93
- skipped_duration: 1607
- skipped_low_content: 314
- skipped_clip_mismatch: 0
- total_hours_before_filtering: 12.3378
- total_hours_after_filtering: 4.5529
- avg_duration: 6.8579
- avg_text_length_chars: 59.19

#### Produced artifacts
- Summary: `project/data/real_run_full_v2/metadata/preprocess_summary.json`
- Clean dataset: `project/data/real_run_full_v2/metadata/dataset.jsonl`
- Skipped audit: `project/data/real_run_full_v2/metadata/skipped.jsonl`
- Duration distribution: `project/data/real_run_full_v2/metadata/duration_distribution.csv`
- HF dataset export was generated during experimentation and then removed during disk cleanup (regenerable from dataset JSONL).

### (b) Baseline vs fine-tuned evaluation on FLEURS-hi
- Status: Baseline complete + improved fine-tune run complete

#### Baseline evaluation pipeline prepared
- Evaluation script: `project/evaluate.py`
- Utility module (normalization + deterministic WER): `project/scripts/asr_eval_utils.py`
- WER was computed after normalizing whitespace and applying consistent tokenization rules to both reference and prediction text.
- Baseline model: `openai/whisper-small`
- Datasets evaluated by script:
  - processed training subset from `project/data/real_run_full_v2/metadata/dataset.jsonl`
  - FLEURS Hindi test (`hi_in`) loaded from direct HF files (`test.tsv` + `audio/test.tar.gz`)

#### Smoke run completed (sanity only)
- Command used:

```powershell
C:/Users/HP/AppData/Local/Programs/Python/Python314/python.exe project/evaluate.py \
  --model_name_or_path openai/whisper-small \
  --processed_jsonl "project/data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_dir project/results/eval_baseline_smoke \
  --max_train_subset 8 \
  --max_fleurs_test 8 \
  --batch_size 2
```

- Smoke WER results:
  - processed subset (n=8): 1.103139
  - FLEURS-hi test (n=8): 0.616667

#### Smoke output artifacts
- Temporary smoke artifacts were removed during workspace cleanup.
- Smoke metrics above are retained in this report for traceability.

#### Full baseline completed on GPU
- Environment/device:
  - torch: `2.10.0+cu128`
  - device: `cuda`
  - gpu: `NVIDIA GeForce RTX 3050 Laptop GPU`

- Command used:

```powershell
c:/projects/josh talks/.venv/Scripts/python.exe project/evaluate.py \
  --model_name_or_path openai/whisper-small \
  --processed_jsonl "project/data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_dir project/results/eval_baseline_full \
  --max_train_subset 200 \
  --max_fleurs_test -1 \
  --batch_size 8
```

- Final baseline WER results:
  - processed subset (n=200): 1.557269
  - FLEURS-hi test (n=418): 0.582006

- Full baseline output artifacts:
  - `project/results/eval_baseline_full/evaluation_summary.json`
  - `project/results/eval_baseline_full/wer_summary.csv`
  - `project/results/eval_baseline_full/predictions_processed_subset.jsonl`
  - `project/results/eval_baseline_full/predictions_fleurs_hi_test.jsonl`

#### Compact fine-tuning run completed on GPU
- Purpose:
  - produce a fast fine-tuned checkpoint under storage/time constraints
  - validate end-to-end fine-tune/evaluate workflow

- Training setup (compact run):
  - max_train_samples: 400
  - max_eval_samples: 40
  - max_steps: 8
  - batch size: 8
  - gradient_accumulation_steps: 1
  - output checkpoint: `project/models/whisper-small-hi-ft`

- Fine-tuned evaluation command:

```powershell
c:/projects/josh talks/.venv/Scripts/python.exe project/evaluate.py \
  --model_name_or_path "project/models/whisper-small-hi-ft" \
  --processed_jsonl "project/data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_dir project/results/eval_finetuned_short \
  --max_train_subset 200 \
  --max_fleurs_test -1 \
  --batch_size 8
```

- Fine-tuned WER results (compact run):
  - processed subset (n=200): 1.718062
  - FLEURS-hi test (n=418): 0.744652

- Fine-tuned output artifacts:
  - `project/results/eval_finetuned_short/evaluation_summary.json`
  - `project/results/eval_finetuned_short/wer_summary.csv`
  - `project/results/eval_finetuned_short/predictions_processed_subset.jsonl`
  - `project/results/eval_finetuned_short/predictions_fleurs_hi_test.jsonl`

- Observation:
  - compact run underfit and underperformed baseline
  - a longer fine-tuning schedule is required for improvement

#### Improved fine-tuning run completed on GPU (final)
- Purpose:
  - run a longer fine-tuning schedule after compact validation
  - target better WER than pretrained baseline
- Rationale:
  - the compact run was used first to validate the training/evaluation pipeline quickly.
  - after observing underfitting in the compact run, train samples and optimization steps were increased to improve generalization.

- Training setup (improved run):
  - initialized from: `project/models/whisper-small-hi-ft`
  - max_train_samples: 1200
  - max_eval_samples: 120
  - max_steps: 30
  - batch size: 8
  - gradient_accumulation_steps: 1
  - learning_rate: 8e-6

- Evaluation command:

```powershell
c:/projects/josh talks/.venv/Scripts/python.exe project/evaluate.py \
  --model_name_or_path "project/models/whisper-small-hi-ft" \
  --processed_jsonl "project/data/real_run_full_v2/metadata/dataset.jsonl" \
  --output_dir project/results/eval_finetuned_v2 \
  --max_train_subset 200 \
  --max_fleurs_test -1 \
  --batch_size 8
```

- Fine-tuned WER results (improved run):
  - processed subset (n=200): 0.821271
  - FLEURS-hi test (n=418): 0.467227

- Improvement vs baseline:
  - FLEURS-hi: 0.582006 -> 0.467227 (absolute -0.114779)
  - processed subset: 1.557269 -> 0.821271 (absolute -0.735998)

- Improved run output artifacts:
  - `project/results/eval_finetuned_v2/evaluation_summary.json`
  - `project/results/eval_finetuned_v2/wer_summary.csv`
  - `project/results/eval_finetuned_v2/predictions_processed_subset.jsonl`
  - `project/results/eval_finetuned_v2/predictions_fleurs_hi_test.jsonl`

### (c) WER report table
| Model | Dataset | WER |
| --- | --- | --- |
| Pretrained Whisper-small (full baseline, GPU) | FLEURS-hi (n=418) | 0.582006 |
| Pretrained Whisper-small (full baseline, GPU) | Processed subset (n=200) | 1.557269 |
| Fine-tuned Whisper-small (compact run) | FLEURS-hi (n=418) | 0.744652 |
| Fine-tuned Whisper-small (compact run) | Processed subset (n=200) | 1.718062 |
| Fine-tuned Whisper-small (improved run, final) | FLEURS-hi (n=418) | 0.467227 |
| Fine-tuned Whisper-small (improved run, final) | Processed subset (n=200) | 0.821271 |

### (d) Systematic sample of 25+ remaining errors
- Status: Complete

- Method:
  - Used final fine-tuned FLEURS predictions: `project/results/eval_finetuned_v2/predictions_fleurs_hi_test.jsonl`
  - Computed per-sample WER with the same normalization/alignment logic as evaluation utilities.
  - Selected the top 30 highest-WER errors (systematic worst-case sample).

- Outputs:
  - `project/results/error_analysis_v2/sample_30_errors.jsonl`
  - `project/results/error_analysis_v2/sample_30_errors.csv`
  - `project/results/error_analysis_v2/analysis_summary.json`

- Sample size summary:
  - total predictions analyzed: 418
  - error samples found: 418
  - systematic sample exported: 30

### (e) Error taxonomy from sampled errors
- Status: Complete

- Taxonomy source file:
  - `project/results/error_analysis_v2/error_taxonomy_counts.csv`

- Category breakdown:
  - substitution_dominant: 356 (85.17%)
  - repetition_hallucination: 48 (11.48%)
  - deletion_dominant: 13 (3.11%)
  - insertion_dominant: 1 (0.24%)

- Example categories and explanations:
  - substitution_dominant:
    - id `1957`: many phonetic near-miss substitutions in Hindi words.
    - id `1860`: mostly wrong lexical choices while sentence structure is preserved.
    - id `1821`: broad token-level substitutions with little insertion/deletion.
  - repetition_hallucination:
    - id `1839`: repeated filler-like loop (`अजी ...`) dominates output.
    - id `2006`: repeated syllabic bursts replace semantic content.
    - id `1760`: long repeated fragments with poor semantic grounding.
  - deletion_dominant:
    - id `1928`: model outputs short tail phrase and drops most reference content.
    - id `1687`: prediction is highly truncated versus reference.
    - id `1995`: only a compact fragment of a long sentence is retained.

### (f) Top-3 frequent error types and fixes
- Status: Complete

- Top-3 frequent error types:
  - substitution_dominant
  - repetition_hallucination
  - deletion_dominant

- Fix candidates:
  - substitution_dominant:
    - increase effective training duration and add more domain speech coverage.
    - tune decoding (beam search + length/repetition penalties).
  - repetition_hallucination:
    - post-process with consecutive-token deduplication.
    - decoding-time anti-repeat constraints (e.g., n-gram repeat control).
  - deletion_dominant:
    - relax decoding truncation risk (generation length and early-stop behavior).
    - improve long-utterance robustness through curriculum/longer segment exposure.

### (g) Implement one fix and show before/after on targeted subset
- Status: Complete

- Implemented fix:
  - consecutive repeated-token deduplication on normalized prediction text.

- Targeted subset:
  - repetition_hallucination samples from the systematic error set.
  - subset size: 11

- Before/after (targeted subset WER):
  - before: 1.520661
  - after: 0.929752
  - delta: -0.590909

- Outputs:
  - `project/results/error_analysis_v2/fix_repetition_target_subset.jsonl`
  - `project/results/error_analysis_v2/analysis_summary.json`

- Reproducibility:
  - analysis script: `project/analyze_errors.py`

#### Integrated into evaluation path (whole-test impact)
- Implementation:
  - Added optional decoding post-process flag in evaluation script:
    - `--dedupe_consecutive_words`
  - Script location:
    - `project/evaluate.py`

- Full-test comparison using final fine-tuned model (`project/models/whisper-small-hi-ft`):
  - without dedup:
    - processed subset (n=200): 0.821271
    - FLEURS-hi (n=418): 0.467227
  - with integrated dedup (`--dedupe_consecutive_words`):
    - processed subset (n=200): 0.609503
    - FLEURS-hi (n=418): 0.451988

- Whole-test absolute improvement from integrated fix:
  - processed subset: -0.211768
  - FLEURS-hi: -0.015239

- Output artifacts (integrated fix run):
  - `project/results/eval_finetuned_v2_dedup/evaluation_summary.json`
  - `project/results/eval_finetuned_v2_dedup/wer_summary.csv`
  - `project/results/eval_finetuned_v2_dedup/predictions_processed_subset.jsonl`
  - `project/results/eval_finetuned_v2_dedup/predictions_fleurs_hi_test.jsonl`

## Q2
- Status: Complete

### Raw ASR generation setup
- Used pretrained model (before fine-tuning): `openai/whisper-small`
- Data source: full processed training set (`project/data/real_run_full_v2/metadata/dataset.jsonl`)
- Generated raw predictions paired with human references for all 2390 segments.
- Script: `project/q2_generate_raw_asr.py`
- Output: `project/results/q2/raw_asr_pretrained.jsonl`

### Cleanup pipeline implemented
- Script: `project/q2_cleanup_pipeline.py`
- Pipeline stages:
  - (a) Number normalization: Hindi number words -> digits using a rule-based parser (supports simple + many compound forms + multipliers like सौ/हजार/लाख/करोड़).
  - (b) English word detection/tagging: tokens marked with `[EN]...[/EN]` when detected as English (Roman script or known Devanagari transliterated English hints).
- Limitation note:
  - English detection is heuristic-based and can miss rare transliterations or occasionally tag ambiguous tokens incorrectly.

### Q2 summary metrics
- total raw ASR pairs processed: 2390
- samples with number normalization changes: 102
- samples with English-word detections: 10
- summary file: `project/results/q2/q2_summary.json`

### (a) Number normalization examples (actual data)
- id `990175_21`:
  - before: `... लगजी एक जो फैमली ...`
  - after: `... लगजी 1 जो फैमली ...`
- id `526266_36`:
  - before: `... कर ले एक साथ ...`
  - after: `... कर ले 1 साथ ...`
- id `520199_37`:
  - before: `... के एक ... मिलते है दो ...`
  - after: `... के 1 ... मिलते है 2 ...`
- id `merged_542785_21_542785_22`:
  - before: `... डो तीन लोग ...`
  - after: `... डो 3 लोग ...`
- id `494019_39`:
  - before: `... में दो वैसे ...`
  - after: `... में 2 वैसे ...`

### Edge-case judgment calls (actual data)
- id `520199_37`:
  - text contains `दो डो नों`; only clear canonical number token `दो` was converted.
  - judgment: keep noisy token `डो` unchanged to avoid over-normalizing ASR noise.
- id `merged_542785_21_542785_22`:
  - text contains `डो तीन`; converted `तीन -> 3` but retained `डो`.
  - judgment: probable misspelling/ASR artifact, not safely mappable to a numeric value.
- id `494019_39`:
  - text contains `तीशेट` near number-like context.
  - judgment: treated as lexical token (not a number phrase), so no conversion.
- Rule note:
  - idiom-protection rules (`दो-चार`, `एक-दो`, etc.) were implemented; no strong idiom-pattern hit appeared in this ASR output slice.

### (b) English word detection examples (actual data)
- id `351501_24`:
  - input fragment: `... प्लान बनाना ...`
  - tagged: `... [EN]प्लान[/EN] बनाना ...`
- id `635909_2`:
  - input: `Hello Hello`
  - tagged: `[EN]Hello[/EN] [EN]Hello[/EN]`
- id `merged_888331_68_888331_69`:
  - input fragment: `... रोल मोडल ...`
  - tagged: `... रोल [EN]मोडल[/EN] ...`

### Q2 output artifacts
- `project/results/q2/raw_asr_pretrained.jsonl`
- `project/results/q2/q2_cleaned_outputs.jsonl`
- `project/results/q2/q2_number_examples.jsonl`
- `project/results/q2/q2_edge_candidates.jsonl`
- `project/results/q2/q2_summary.json`

## Q3
- Status: Complete (using provided external file)

### Input used
- External file provided: `project/data/Unique Words Data - Sheet1.csv`
- Word column used: `word`

### (a) Correct vs incorrect spelling classification approach
- Script: `project/q3_word_quality.py`
- Built target vocabulary from external unique-word file and merged available corpus frequency signals from `project/data/real_run_full_v2/metadata/dataset.jsonl` where possible.
- Classification rules combine:
  - script checks (Devanagari-heavy vs noisy/non-Devanagari)
  - suspicious character repetition patterns
  - corpus frequency priors (high frequency -> likely correct)
  - rare long-token risk heuristics

### (b) Confidence scoring + reason
- Each word includes:
  - `classification` (`correct spelling` / `incorrect spelling`)
  - `confidence` (`high` / `medium` / `low`)
  - short `reason`
- Output file with reasons:
  - `project/results/q3_external/q3_word_classification.csv`

### (c) Low-confidence review (40-50 words)
- Reviewed: 50 low-confidence words
- Right: 41
- Wrong: 9
- File: `project/results/q3_external/q3_low_confidence_review.csv`
- Interpretation:
  - Approach is useful for large-scale triage.
  - Most failures occur on transliterated/borrowed forms and rare proper nouns where orthography varies naturally.

### (d) Unreliable categories (at least 1-2)
- Transliteration-heavy borrowed words and hybrid tokens:
  - e.g., `एक्सपीरिएंसिस`, `ड्राविडियन्स`, `कुटुम्बकम्` (classification confidence drops due to orthographic variants).
- Rare proper nouns / low-context forms:
  - low-frequency entities often look like misspellings without surrounding sentence context.
- Additional insight:
  - frequency-based heuristics are strong for common words but degrade on low-frequency and morphologically complex tokens.

### Q3 deliverables
- Final unique correct-spelling count (external file run): **170556**
- Full classification + confidence + reason:
  - `project/results/q3_external/q3_word_classification.csv`
- Requested 2-column sheet-ready file (`word`, `classification`):
  - `project/results/q3_external/q3_google_sheet_ready.csv`
- Summary:
  - `project/results/q3_external/q3_summary.json`

## Q4
- Status: Complete

### Design choice: alignment unit
- Chosen unit: **word-level**
- Rationale:
  - WER is word-based and assignment asks lattice-adjusted WER.
  - Word bins are interpretable for lexical/phonetic/spelling alternatives.

### Lattice construction approach (theory)
- Inputs: 5 model outputs for same utterance + human reference.
- For each utterance:
  - Start with reference words as sequential bins.
  - Align each model hypothesis to reference via edit-distance backtrace.
  - Add aligned model alternatives into bins.
  - Trust model agreement over reference when alternative token frequency >= threshold (used: 3 of 5).
  - Threshold rationale: a majority threshold of 3 was chosen to improve robustness while reducing the chance of overfitting to noisy model-specific outputs.
  - For insertion-heavy zones, create optional bins (including epsilon `""`) when multiple models agree.
- Lattice-aware scoring:
  - substitution cost = 0 if token belongs to current bin alternatives else 1.
  - deletion cost = 0 for optional epsilon bins, else 1.
  - insertion cost = 1.

### Implementation
- Script: `project/q4_from_task_csv.py`
- Pseudocode/code is embedded in the script and fully runnable.

### Data used
- Provided file: `project/data/Question 4 - Task.csv`
- Columns used:
  - reference: `Human`
  - models: `Model H`, `Model i`, `Model k`, `Model l`, `Model m`, `Model n`

### Lattice-based WER results
- Samples evaluated: 46
- Summary file:
  - `project/results/q4_task/q4_task_lattice_summary.csv`

| Model | Standard WER | Lattice WER | Delta |
| --- | --- | --- | --- |
| Model H | 0.024450 | 0.019560 | -0.004890 |
| Model i | 0.003667 | 0.003667 | 0.000000 |
| Model k | 0.063570 | 0.056235 | -0.007335 |
| Model l | 0.055012 | 0.050122 | -0.004890 |
| Model m | 0.123472 | 0.113692 | -0.009780 |
| Model n | 0.077017 | 0.067237 | -0.009780 |

### Interpretation
- Lattice evaluation reduced unfair penalties for 5/6 models and kept 1 model unchanged (Model i), matching the assignment expectation.
- This indicates consensus-driven alternatives helped where reference/model mismatch existed, without artificially boosting already well-aligned outputs.

### Q4 artifacts
- `project/results/q4_task/q4_task_lattice_summary.csv`
- `project/results/q4_task/q4_task_lattice_summary.json`
- `project/results/q4_task/q4_task_lattice_preview.json`

## Notes for final submission pack
- Include this file as narrative report.
- Include all generated JSON/CSV outputs under `project/results` and `project/data/real_run_full_v2/metadata`.
- Core model artifact is in `project/models/whisper-small-hi-ft`.
- Q3 external-file deliverables are in `project/results/q3_external`.
- Q4 provided-task deliverables are in `project/results/q4_task`.

## Final Conclusion
This project demonstrates that data quality improvements and targeted post-processing can significantly improve ASR performance even with limited fine-tuning.

Model fine-tuning produced substantial gains, and error-driven fixes such as repetition handling further improved robustness on both targeted and full-test evaluations.

Future work will focus on decoding optimization and stronger domain-specific adaptation for harder lexical and long-utterance cases.
