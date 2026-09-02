# Output Guidelines

## Data Persistence

**Dump large/structured data to `data/` directory, return only summaries.**

When tool returns data, check if it exceeds thresholds:
- Lists/dicts with >10 items → dump to `data/`
- Text >500 characters → dump to `data/`
- Tabular data >5 rows → dump to `data/`

Always return: storage path, record count, schema/structure, optional sample (1-2 items).

## Output Format Requirements

ALWAYS save final deliverables to `results/` directory. Choose format based on data type:
- **JSON**: API responses, nested data, complex objects (most versatile)
- **CSV**: Tabular data, time series, lists with consistent fields
- **Markdown**: Reports, summaries, documentation

**Example patterns:**
```python
import json

# API response -> JSON (preserves structure for reprocessing)
result = some_tool(param="value")
with open('results/api_response.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Saved API response to results/api_response.json")

# Tabular data -> CSV (easy to load with pandas later)
df.to_csv('results/data_output.csv', index=False)
print(f"Saved {len(df)} rows to results/data_output.csv")

# Report/summary -> Markdown
report = f"""# Analysis Report
## Summary
{summary_text}
## Data
{formatted_data}
"""
with open('results/analysis_report.md', 'w') as f:
    f.write(report)
print("Saved report to results/analysis_report.md")
```

**IMPORTANT**: All files intended for user consumption MUST go in `results/` directory.
Always print the output path so the user knows where to find results.
