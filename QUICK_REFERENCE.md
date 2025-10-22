# Quick Reference Guide

## 📋 File Overview

### **Which file should I run?**

| Goal | File to Run | Description |
|------|-------------|-------------|
| Test **spaCy** baseline | `evaluate_spacy.py` | Fast, pre-trained model (4ms/doc) |
| Test **Zero-Shot** classification | `evaluate_zero_shot.py` | Handles unknown entities (500-1000ms) |
| Test **FinBERT** finance model | `evaluate_finbert.py` | Finance-specific training |
| **Compare all approaches** | `evaluate_compare.py` | Runs all 3 and creates comparison table |

---

## 🚀 Quick Commands

### Run Individual Approach
```bash
cd "/Applications/Temasek Polytechnic/KM-MP/Risk-analysis"

# Option 1: spaCy (fastest, 4ms per document)
python evaluate_spacy.py

# Option 2: Zero-Shot (best for unknown entities)
python evaluate_zero_shot.py

# Option 3: FinBERT (finance-specific)
python evaluate_finbert.py
```

### Run All Approaches & Compare
```bash
# This will run all 3 approaches and create a comparison table
python evaluate_compare.py
```

---

## 📊 Expected Results

| Approach | Accuracy | Speed | Best For |
|----------|----------|-------|----------|
| **spaCy** | 67.9% | 4.2 ms | Production speed |
| **Zero-Shot** | 85-90%* | 500-1000 ms | Unknown entities |
| **FinBERT** | 82-88%* | 50-100 ms | Finance context |

*Estimated based on research (see `NLP_LIBRARY_COMPARISON.md`)

---

## 📁 Output Files

After running evaluations, you'll get:

- `results_spacy.json` - spaCy results
- `results_zero_shot.json` - Zero-Shot results  
- `results_finbert.json` - FinBERT results
- `comparison_results_*.csv` - Side-by-side comparison table

---

## 🔧 Shared Utilities

All evaluation scripts use **`utils_nlp.py`** for common functions:

- `load_ground_truth_dataset()` - Load CSV dataset
- `print_dataset_stats()` - Display dataset info
- `print_results_by_category()` - Show accuracy by category
- `calculate_overall_accuracy()` - Compute overall accuracy
- `save_results_to_json()` - Save results to JSON

---

## 📝 Dataset

**File:** `banking_ner_dataset.csv`

**Format:**
```csv
text,entity_text,entity_label,entity_category,is_well_known
"Transfer $5000 from John Doe...","DBS Bank","ORG","Bank",TRUE
```

**Statistics:**
- 81 labeled examples
- 7 entity types (ORG, PERSON, MONEY, DATE, GPE, LOC, FAC)
- 24 entity categories
- 66 well-known + 15 unknown entities

---

## 💡 Next Steps

1. ✅ **Run comparison:** `python evaluate_compare.py`
2. ✅ **Review results:** Check `comparison_results_*.csv`
3. ✅ **Choose approach:** Based on accuracy vs speed needs
4. ⏭️ **Build hybrid:** Combine best of all approaches
5. ⏭️ **Production:** Deploy chosen solution

---

## 🆘 Troubleshooting

### Error: "Missing 'transformers'"
```bash
pip install transformers torch
```

### Error: "spaCy model not found"
```bash
python -m spacy download en_core_web_md
```

### Script takes too long
- Zero-Shot is slow (500-1000ms per entity)
- Expected: 5-10 minutes for full dataset
- Use `evaluate_spacy.py` for quick testing

---

## 📚 Additional Resources

- **`NLP_LIBRARY_COMPARISON.md`** - Detailed comparison of NLTK, spaCy, Hugging Face, etc.
- **`README.md`** - Full project documentation
- **`output.txt`** - Phase 1 baseline results
