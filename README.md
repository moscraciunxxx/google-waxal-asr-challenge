# Google WAXAL ASR Challenge

Open-source ASR experiments and submission tooling for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge).

This publication contains the Python source, validation tests, and one curated public submission artifact. Large models, audio, reconstructed datasets, experiment outputs, and virtual environments are intentionally not versioned.

## Curated submission

The latest public Phase-2 candidate is:

`submission/phase2_public_final.csv`

It has the exact `ID,Target` schema and 2,392 expanded Phase-2 rows. The latest measured public score before the final text-normalization pass was `0.707243556`; the final pass applies a locked-validation-backed removal of terminal Luganda filler `aa` tokens. This is an artifact for publication and upload, not a guarantee of leaderboard rank.

Validate it with:

```bash
python scripts/validate_public_submission.py
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/ -q
```

To reconstruct the public WAXAL tables locally:

```bash
python scripts/prepare_data.py
```

The ASR pipeline uses Hugging Face `google/WaxalNLP`, open pretrained models, PyTorch, Transformers, and `jiwer`. Run `make data`, `make train-smoke`, `make eval`, `make predict`, or `make submit` for the baseline workflow. Full training and decoding require downloaded datasets, audio, and model weights; those assets are excluded from Git.

## Rules and evaluation

- Only open-source models and datasets are used.
- Phase-1 test transcripts are blocked from training and tuning by `FORBIDDEN_TRAIN_SPLITS = {"test"}`.
- Phase 2 is decoded from audio; metadata is not required by `src/infer.py`.
- The local metric is `0.5 * WER + 0.5 * CER`; Zindi displays the complementary higher-is-better score.
- The fixed reproducibility seed is `42`.

## Layout

```text
src/          reusable training, decoding, normalization, metrics, and submission code
scripts/      data preparation, evaluation, decoding, and candidate-building entry points
tests/        unit and structural tests
submission/  curated public submission and its publication notes
data/         generated locally; only data/README.md is tracked
```

## Models and data disclosure

- WAXAL ASR dataset: https://huggingface.co/datasets/google/WaxalNLP
- Whisper: https://huggingface.co/openai/whisper-small
- MMS: https://huggingface.co/facebook/mms-1b-all

Check the upstream licenses before redistributing downloaded model weights or competition data. Credentials belong in an untracked `.env` file; `.env.example` documents the optional variables.
