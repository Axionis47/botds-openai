# Bot Data Scientist (OpenAI-Led Decisions)

A local-first, deterministic bot data scientist for static datasets that runs on Mac M4 16GB. **OpenAI** is the sole decision authority for all critical choices, with optional Llama (Ollama) for non-critical drafting only.

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ollama for local Llama (optional, for non-critical drafting)
brew install ollama
ollama serve
ollama pull llama3.2
```

### 2. Set OpenAI API Key (Required)

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

**Critical:** If `OPENAI_API_KEY` is missing, the system will fail fast with a clear error. No fallback to other models for critical decisions.

### 3. Run Analysis

```bash
# Run on built-in datasets
python -m cli.run --config configs/iris.yaml
python -m cli.run --config configs/breast_cancer.yaml  
python -m cli.run --config configs/diabetes.yaml

# Run on your own CSV
python -m cli.run --config configs/csv_template.yaml
```

### 4. View Results

Artifacts are written to `artifacts/<job_id>/`:
- `one_pager.html` - Executive summary
- `appendix.html` - Technical details
- `handoffs/` - Stage-by-stage outputs
- `logs/decision_log.jsonl` - All critical decisions by OpenAI

## Authority Policy

**Critical Decisions (OpenAI Only):**
1. Target validity & task type confirmation
2. Split policy selection & leakage denial list
3. Feature inclusion/exclusion when risky
4. Model selection when metrics are tied/close
5. Operating threshold policy
6. Fairness override decisions
7. Final report sign-off text

**Non-Critical Drafting (Llama OK):** Intermediate bullet drafts, note formatting. These cannot change gating outcomes.

## Service Boundaries & Handoffs

| Stage | Schema | Output File | Purpose |
|-------|--------|-------------|---------|
| Profiling | `profile.schema.json` | `handoffs/profile.json` | Dataset shape, types, missingness |
| EDA | `eda.schema.json` | `handoffs/eda.json` | Top insights & hypotheses |
| Features | `feature_plan.schema.json` | `handoffs/feature_plan.json` | Transform plan & rationale |
| Splits | `split_indices.schema.json` | `handoffs/split_indices.json` | Train/val/test indices |
| Modeling | `ladder.schema.json` | `handoffs/ladder.json` | Model leaderboard with CI |
| Evaluation | `evaluation.schema.json` | `handoffs/evaluation.json` | Test metrics & robustness |
| Reporting | `report.schema.json` | `handoffs/report.json` | Final report payload |

## Cache Strategy

Cache key = hash of (dataset + schema + task + context). Modes:
- `warm`: Reuse cached results
- `paranoid`: Recompute metrics, reuse heavy aggregates  
- `cold`: Ignore cache entirely

## Common Issues

See `RUNBOOK.md` for troubleshooting:
- Missing OpenAI API key
- Leakage detection blocks
- Budget limit breaches
- Cache invalidation

## Quick Test

To verify the system works:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# Install dependencies
pip install -r requirements.txt

# Run system test
python test_system.py
```

## Architecture

- **OpenAI (gpt-4o-mini)**: All critical decisions via function calling
- **Ollama (llama3.2)**: Optional non-critical drafting only
- **Handoff artifacts**: JSON files matching schemas for full traceability
- **Budget guards**: Time/memory/token checkpoints at each stage
- **Cache system**: Intelligent invalidation based on input changes

## Testing

Run the full test suite:

```bash
# Install test dependencies
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_cross_dataset.py -v
python -m pytest tests/test_cache_and_invalidation.py -v
python -m pytest tests/test_acceptance_suite.py -v
```
