# Bot Data Scientist Runbook

This runbook covers common issues and their resolutions.

## Common Failures

### 1. Missing OpenAI API Key

**Error:**
```
ValueError: OPENAI_API_KEY environment variable is required. OpenAI is the sole decision authority - no fallback available.
```

**Resolution:**
1. Get an OpenAI API key from https://platform.openai.com/api-keys
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```
3. Or add it to your `.env` file:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### 2. Leakage Detection Block

**Error:**
```
RuntimeError: Data quality check failed: [{'column': 'perfect_predictor', 'issue': 'perfect_correlation', 'correlation': 1.0, 'severity': 'block'}]
```

**Resolution:**
1. Review the offending columns in the error message
2. Remove or modify the problematic columns from your dataset
3. If the correlation is legitimate, you can add the column to the deny list in your config:
   ```yaml
   # This would require manual config editing - the system blocks automatically
   ```
4. For CSV data, clean the data before running the pipeline

### 3. Budget Limit Breach

**Error:**
```
RuntimeError: Budget exceeded at stage model_ladder: ['Time budget exceeded - consider aborting']
```

**Resolution:**
1. Increase budget limits in your config:
   ```yaml
   budgets:
     time_min: 45  # Increase from 25
     memory_gb: 8  # Increase from 4
     token_budget: 12000  # Increase from 8000
   ```
2. Or accept shortcuts by allowing the system to continue with reduced functionality
3. For large datasets, consider sampling:
   ```yaml
   sampling:
     eda_rows: 50000  # Reduce from 200000
   ```

### 4. Ollama Connection Issues

**Error:**
```
Warning: Ollama not available for non-critical drafting
```

**Resolution:**
This is not a fatal error. Ollama is optional for non-critical drafting only.

To fix if desired:
1. Install Ollama: `brew install ollama`
2. Start the service: `ollama serve`
3. Pull the model: `ollama pull llama3.2`
4. Check the base URL in your config:
   ```yaml
   # In .env file
   OLLAMA_BASE_URL=http://localhost:11434
   ```

### 5. CSV File Not Found

**Error:**
```
FileNotFoundError: CSV file not found: path/to/your/data.csv
```

**Resolution:**
1. Check the file path in your config:
   ```yaml
   data:
     csv_paths: 
       - "correct/path/to/your/data.csv"
   ```
2. Use absolute paths if relative paths don't work
3. Ensure the file exists and is readable

### 6. Invalid Target Column

**Error:**
```
ValueError: Target column 'wrong_name' not found
```

**Resolution:**
1. Check your CSV file column names
2. Update the target in your config:
   ```yaml
   data:
     target: "correct_target_column_name"
   ```
3. For built-in datasets, leave target empty (auto-detected)

## Performance Issues

### Slow Pipeline Execution

**Symptoms:**
- Pipeline takes longer than expected
- High memory usage
- Frequent budget warnings

**Solutions:**
1. **Reduce dataset size:**
   ```yaml
   sampling:
     eda_rows: 10000  # Sample for EDA
   ```

2. **Use simpler models:**
   - The system will automatically choose appropriate models
   - Tight budgets trigger simpler model selection

3. **Enable caching:**
   ```yaml
   cache:
     mode: "warm"  # Reuse previous results
   ```

4. **Increase budgets:**
   ```yaml
   budgets:
     time_min: 60
     memory_gb: 8
   ```

### High Token Usage

**Symptoms:**
- Token budget exceeded warnings
- Expensive OpenAI API bills

**Solutions:**
1. **Use smaller model:**
   ```yaml
   llms:
     openai_model: "gpt-4o-mini"  # Already the default
   ```

2. **Increase token budget:**
   ```yaml
   budgets:
     token_budget: 15000
   ```

3. **Enable caching to avoid re-decisions:**
   ```yaml
   cache:
     mode: "warm"
   ```

## Data Issues

### High PII Risk

**Warning:**
```
PII scan found high risk: 150 email addresses detected
```

**Resolution:**
1. **Enable redaction (default):**
   ```yaml
   pii:
     enabled: true
     action: "redact"  # Replaces PII with [REDACTED]
   ```

2. **Block processing if PII is unacceptable:**
   ```yaml
   pii:
     action: "block"  # Stops pipeline if PII found
   ```

3. **Clean data before processing:**
   - Remove PII columns manually
   - Use data anonymization tools

### Imbalanced Dataset

**Issue:** Severe class imbalance affecting model performance

**Solutions:**
1. **Enable stratified sampling:**
   ```yaml
   sampling:
     stratify_by: ["target_column"]
   ```

2. **Use appropriate metrics:**
   ```yaml
   metrics:
     primary: "pr_auc"  # Better for imbalanced data than accuracy
   ```

3. **The system will automatically handle class imbalance in model selection**

## Debugging

### Enable Verbose Output

```bash
python -m cli.run --config configs/your_config.yaml --verbose
```

### Check Logs

All logs are saved in `artifacts/<job_id>/logs/`:
- `decision_log.jsonl` - All OpenAI decisions
- `handoff_ledger.jsonl` - Stage-to-stage data flow
- Pipeline prints progress to console

### Inspect Intermediate Results

All intermediate results are saved in `artifacts/<job_id>/handoffs/`:
- `profile.json` - Dataset analysis
- `split_indices.json` - Train/val/test splits
- `feature_plan.json` - Feature engineering plan
- `ladder.json` - Model comparison results

### Cache Debugging

Check cache status:
```yaml
cache:
  mode: "paranoid"  # Recompute metrics but reuse heavy operations
```

Clear cache:
```bash
rm -rf ./cache/*
```

## Getting Help

1. **Check this runbook first**
2. **Review the error message carefully** - they contain specific guidance
3. **Check the decision log** in `artifacts/<job_id>/logs/decision_log.jsonl` for OpenAI's reasoning
4. **Try with a smaller dataset** to isolate the issue
5. **Verify your OpenAI API key has sufficient credits**

## Emergency Procedures

### Pipeline Stuck/Hanging

1. **Kill the process:** Ctrl+C
2. **Check budget limits** - may be waiting for OpenAI response
3. **Verify OpenAI API key is valid**
4. **Try with tighter budgets to force faster execution**

### Out of Memory

1. **Reduce dataset size:**
   ```yaml
   sampling:
     eda_rows: 5000
   ```
2. **Increase memory budget:**
   ```yaml
   budgets:
     memory_gb: 16
   ```
3. **Use cold cache to free memory:**
   ```yaml
   cache:
     mode: "cold"
   ```

### API Rate Limits

**Error:** OpenAI API rate limit exceeded

**Solutions:**
1. **Wait and retry** - rate limits reset over time
2. **Upgrade OpenAI plan** for higher limits
3. **Use caching** to reduce API calls:
   ```yaml
   cache:
     mode: "warm"
   ```
