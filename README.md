# superstoreAnalysis

Business Intelligence & Data Mining solution for the Global Superstore dataset.

## Deliverables included

- `superstore_analysis.py`  
  End-to-end analysis script that reads `Global-Superstore.csv` and generates:
  - Interactive dashboard (`output/interactive_dashboard.html`)
  - Analytical report (`output/analytical_report.md`)
  - Data mining findings (`output/data_mining_findings.md`)
  - Strategic recommendations (`output/strategic_recommendations.md`)

## Run

```bash
python superstore_analysis.py --input /path/to/Global-Superstore.csv --output-dir output
```

## Required Python packages

- pandas
- plotly