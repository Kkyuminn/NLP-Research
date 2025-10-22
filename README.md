# Entity Classification for Banking Metadata Extraction

## 🎯 Project Goal
Research and evaluate NLP approaches for extracting and classifying entities in banking documents to generate rich metadata for a central indexing hub.

**Success Criteria:**
- ≥85% entity extraction accuracy
- ≥80% classification accuracy
- Handle unknown entities with ≥70% confidence
- Process <5 seconds per document

## 📂 Project Structure

```
Risk-analysis/
│
├── 🔬 EVALUATION SCRIPTS (One per NLP approach)
│   ├── evaluate_spacy.py          # spaCy baseline - Fast (4ms)
│   ├── evaluate_zero_shot.py      # Zero-Shot Classification - Handles unknowns (500-1000ms)
│   ├── evaluate_finbert.py        # FinBERT - Finance-specific (50-100ms)
│   └── evaluate_compare.py        # Run all approaches and compare results
│
├── 🛠️ SHARED UTILITIES
│   └── utils_nlp.py               # Common functions for all evaluation scripts
│
├── 📊 DATASET
│   └── banking_ner_dataset.csv    # 81 labeled examples (7 entity types)
│
├── 📚 DOCUMENTATION
│   ├── README.md                  # This file - Project overview
│   ├── QUICK_REFERENCE.md         # Command cheatsheet
│   └── NLP_LIBRARY_COMPARISON.md  # Detailed library comparison
│
├── 📦 ARCHIVE
│   ├── archive/                   # Different use case datasets
│   └── _archive_legacy/           # Legacy code (reference only)
│
└── 📁 OUTPUT (generated after running)
    ├── results_spacy.json
    ├── results_zero_shot.json
    ├── results_finbert.json
    └── comparison_results_*.csv
```

## 📊 Phase 1 Results (spaCy Baseline)

| Metric | Result | Status |
|--------|--------|--------|
| Entity Extraction Accuracy | 93.8% | ✅ Excellent |
| Entity Classification Accuracy | 67.9% | ⚠️ Needs improvement |
| Well-known entities | 66.7% | ⚠️ Needs improvement |
| Unknown/local entities | 73.3% | ⚠️ Needs improvement |
| Processing Speed | 4.2 ms/doc | ✅ Production-ready |
| Throughput | 237 docs/sec | ✅ Fast |

**Key Findings:**
- ✅ **Strengths:** Very fast, good at extraction, 100% accuracy on banks/hospitals/universities
- ❌ **Weaknesses:** Poor on unknown entities, low accuracy on e-commerce (0%), money (28.6%), dates (50%)

## 🚀 Next Steps

### Phase 2: Advanced NLP Approaches ⏭️
- [ ] Run Zero-Shot Classification evaluation
- [ ] Run FinBERT evaluation  
- [ ] Compare all approaches side-by-side
- [ ] Analyze accuracy vs speed trade-offs

### Phase 3: Hybrid System (Production) ⏭️
- [ ] Build hybrid metadata extraction pipeline:
  - spaCy for fast extraction (93.8% found)
  - Zero-Shot for unknown entity classification
  - Knowledge base for top 100 frequent entities
  - **Expected:** 88-92% overall accuracy, 50-100ms speed

## 💡 Key Benefits

✅ **Clean Separation** - Each NLP approach in its own file  
✅ **No Duplication** - Shared utilities in `utils_nlp.py`  
✅ **Easy Testing** - Test any approach independently  
✅ **Easy Comparison** - `evaluate_compare.py` orchestrates all  
✅ **Extensible** - Easy to add new approaches in future

## 🆘 Troubleshooting

**Problem:** `Missing 'transformers'`  
**Solution:** `pip install transformers torch`

**Problem:** `spaCy model not found`  
**Solution:** `python -m spacy download en_core_web_md`

**Problem:** Script takes too long  
**Solution:** Zero-Shot is slow (5-10 min for full dataset). Use `evaluate_spacy.py` for quick testing.

## 📚 Additional Documentation

- **QUICK_REFERENCE.md** - Command cheatsheet and quick guide
- **NLP_LIBRARY_COMPARISON.md** - Detailed comparison of NLTK, spaCy, Hugging Face Transformers, Gensim, Flair, Stanza, and Cloud APIs
- **_archive_legacy/** - Legacy code kept for reference only

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install required packages
pip install transformers torch spacy pandas

# Download spaCy model
python -m spacy download en_core_web_md
```

### 2. Run Individual NLP Approach
```bash
cd "/Applications/Temasek Polytechnic/KM-MP/Risk-analysis"

# Option 1: spaCy (fastest, 4ms per document)
python evaluate_spacy.py

# Option 2: Zero-Shot (best for unknown entities, 500-1000ms)
python evaluate_zero_shot.py

# Option 3: FinBERT (finance-specific, 50-100ms)
python evaluate_finbert.py
```

### 3. Compare All Approaches
```bash
# Run all 3 approaches and create comparison table
python evaluate_compare.py
```

## 📊 Expected Results

| Approach | Accuracy | Speed | Best For |
|----------|----------|-------|----------|
| **spaCy** | 67.9% | 4.2 ms | Production speed ⚡ |
| **Zero-Shot** | 85-90%* | 500-1000 ms | Unknown entities ⭐ |
| **FinBERT** | 82-88%* | 50-100 ms | Finance context 💰 |

*Estimated based on research (see `NLP_LIBRARY_COMPARISON.md`)

## 📁 Output Files

After running evaluations:
- `results_spacy.json` - spaCy detailed results
- `results_zero_shot.json` - Zero-Shot detailed results
- `results_finbert.json` - FinBERT detailed results
- `comparison_results_*.csv` - Side-by-side comparison table

## 📝 Dataset Details

**81 labeled examples covering:**
- 7 entity types: ORG, PERSON, MONEY, DATE, GPE, LOC, FAC
- 24 entity categories (Bank, School, Hospital, Tech, Retail, etc.)
- 66 well-known entities (DBS, NUS, Twitter, Google, etc.)
- 15 unknown/local entities (Breadtalk, PropertyGuru, Koufu, etc.)

**Entity breakdown:**
- 51 Organizations (banks, schools, companies)
- 6 People (customers, CEOs, employees)
- 7 Money amounts (amounts, currencies)
- 6 Dates (transaction dates, periods)
- 7 Geopolitical entities (countries)
- 3 Locations (streets, places)
- 1 Facility (buildings)

