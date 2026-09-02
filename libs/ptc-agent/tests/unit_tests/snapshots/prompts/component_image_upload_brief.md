# Image Auto-Upload

Images saved to `results/` are automatically uploaded to cloud storage. The URL is returned in the response.

**CRITICAL**: Use RELATIVE paths, NOT absolute paths:
- CORRECT: `plt.savefig('results/chart.png')` -> auto-uploaded, URL returned
- WRONG: `plt.savefig('/results/chart.png')` -> NOT uploaded!

Example:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.savefig('results/chart.png')  # Use relative path!
print("Chart saved")
```

The response includes: `![chart](https://storage-url/chart.png)` - reference this URL in your report.
