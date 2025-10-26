# A Multi-View Labeled Dataset of Python Code Review Comments

This dataset contains Python code review comments labeled with categories and subcategories defined in ESEM’23 research

## Data Source
- Crawled from **Top 100 Python repositories on GitHub**

## Taxonomy
**Source**: Adapted from Turzo et al., *Towards Automated Classification of Code Review Feedback to Support Analytics*, ESEM 2023.

Categories: `functional`, `refactoring`, `documentation`, `discussion`, `false positive`.

## Schema
| Field                | Type      | Description |
|----------------------|-----------|-------------|
| `comment`           | string    | Reviewer’s remark |
| `code`              | string    | Diff hunk snippet |
| `file_path`         | string    | File path in PR |
| `pr_number`         | integer   | Pull request ID |
| `repo`, `owner`     | string    | Repository coordinates |
| `comment_id`        | integer   | GitHub comment ID |
| `comment_created_at`| string    | ISO timestamp |
| `subcategory`       | string    | Fine-grained label |
| `category`          | string    | Coarse label |
| `enriched`          | string    | Combined context |
| `line_number`       | integer   | Optional |

## Data
The dataset is split into 5 chunks for easier handling and parallel processing:
```
chunk_01.json (n=3342)
chunk_02.json (n=3342)
chunk_03.json (n=3342)
chunk_04.json (n=3342)
chunk_05.json (n=3343)
Total: 16,711 records
```

### Sample Usage
```python
import json
with open("data/chunk_01.json", "r", encoding="utf-8") as f:
    records = json.load(f)
print(f"Loaded {len(records)} records")
print(records[0])
```

## Methodology
Pipeline:
1. **Crawl**: GitHub API → PR review comments + diffs from Top 100 Python repositories.
2. **Preprocess**: Normalize (lowercase, remove ```suggestion ...``` blocks and backticks), deduplicate on `(code, comment)`, shuffle, and split into balanced chunks.
3. **Label**: Manual seed set + SVM-based multi-view classifier for subcategories; map subcategory → category via a fixed taxonomy.
4. **Train**: Train subcategory model; evaluate per-class metrics and reuse the same artifacts for inference to avoid feature mismatches.

## Referances
Turzo, A.K., Faysal, F., Poddar, O., Sarker, J., Iqbal, A., & Bosu, A. (2023). *Towards Automated Classification of Code Review Feedback to Support Analytics*. In *Proceedings of the 17th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM)*. DOI: 10.48550/arXiv.2307.03852

```bibtex
@dataset{icoz2025crcpy,
  title  = {A Multi-View Labeled Dataset of Python Code Review Comments},
  author = {Busra Icoz},
  year   = {2025},
  url    = {https://github.com/crc-py-dataset}
}
```
