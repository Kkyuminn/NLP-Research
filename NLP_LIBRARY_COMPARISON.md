# NLP Library Comparison for Banking Entity Classification

## 🎯 Your Use Case
Extract and classify entities (organizations, people, money, dates, locations) from banking documents to generate metadata for indexing hub.

---

## 📚 Comprehensive NLP Library Comparison

### **1. spaCy** ⭐⭐⭐ **Currently Testing**

**What it does:** Industrial-strength NLP with pre-trained models for NER, POS tagging, dependency parsing

**Strengths:**
- ✅ Very fast (4.6ms per document in our tests)
- ✅ Pre-trained models ready to use (no training needed)
- ✅ Excellent for entity extraction (94.1% found in our test)
- ✅ Supports 18 entity types (PERSON, ORG, MONEY, DATE, GPE, etc.)
- ✅ Easy to use and integrate
- ✅ Production-ready

**Weaknesses:**
- ❌ Limited accuracy on unknown entities (71.4% on local companies)
- ❌ Not finance-specific (trained on general English text)
- ❌ Can't classify entities it doesn't recognize

**Performance (Our Baseline):**
- Entity Extraction: 94.1%
- Entity Classification: 76.5%
- Speed: 4.6 ms/document
- Throughput: 217 docs/second

**Best For:**
- Fast entity extraction
- Production systems needing high throughput
- General NLP tasks

**Code Example:**
```python
import spacy
nlp = spacy.load('en_core_web_md')
doc = nlp("Transfer $5000 to DBS Bank")
for ent in doc.ents:
    print(ent.text, ent.label_)
# Output: $5000 MONEY, DBS Bank ORG
```

---

### **2. NLTK (Natural Language Toolkit)** ⚠️ **Too Basic**

**What it does:** Educational NLP library with basic text processing tools

**Strengths:**
- ✅ Good for learning NLP concepts
- ✅ Lightweight
- ✅ Lots of text processing utilities (tokenization, stemming, etc.)
- ✅ Free and open-source

**Weaknesses:**
- ❌ **NOT suitable for entity classification** (no good pre-trained NER models)
- ❌ Very basic NER (low accuracy ~40-50%)
- ❌ Slow compared to modern libraries
- ❌ Requires manual feature engineering
- ❌ Not production-ready

**Estimated Performance:**
- Entity Extraction: ~40-60% (poor)
- Entity Classification: ~30-50% (very poor)
- Speed: 50-100 ms/document (slow)

**Best For:**
- Learning NLP basics
- Text preprocessing (tokenization, stemming)
- Academic research
- **NOT for your banking use case!**

**Code Example:**
```python
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
text = "Transfer $5000 to DBS Bank"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)
# Output: Very basic, often inaccurate
```

**Verdict:** ❌ **Skip NLTK** - too basic for entity classification

---

### **3. Hugging Face Transformers** ⭐⭐⭐⭐ **HIGHLY RECOMMENDED**

**What it does:** Access to 100,000+ pre-trained transformer models (BERT, GPT, T5, etc.)

**Strengths:**
- ✅ **State-of-the-art accuracy** (often 85-95%)
- ✅ **Finance-specific models available** (FinBERT, BankBERT)
- ✅ **Zero-Shot Classification** (can classify unknown entities!)
- ✅ Supports all entity types + custom categories
- ✅ Easy to fine-tune on your data
- ✅ Huge model library

**Weaknesses:**
- ⚠️ Slower than spaCy (50-1000ms depending on model)
- ⚠️ Requires more memory (GPU recommended)
- ⚠️ More complex setup

**Estimated Performance:**
- Entity Extraction: 90-95% (excellent)
- Entity Classification: 85-92% (excellent)
- **Unknown entities: 85-90%** ⭐ (best in class!)
- Speed: 50-1000 ms/document (varies by model)

**Best For:**
- **Your banking use case!** ✅
- Handling unknown entities (Zero-Shot)
- Finance-specific NLP (FinBERT)
- State-of-the-art accuracy

**Recommended Models for Your Project:**

#### **A. Zero-Shot Classification** (Handles Unknown Entities)
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", 
                     model="facebook/bart-large-mnli")

entity = "Breadtalk Group"  # Unknown entity
labels = ["bank", "school", "tech company", "restaurant", "retail"]
result = classifier(entity, candidate_labels=labels)
# Output: "restaurant" or "retail" (correct!)
```
- **Estimated Accuracy:** 85-90%
- **Speed:** 500-1000 ms/document
- **Best Feature:** Works on ANY entity, even never seen before!

#### **B. FinBERT** (Finance-Specific)
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner = pipeline("ner", model=model, tokenizer=tokenizer)

text = "Transfer to DBS Bank account"
entities = ner(text)
# Better at recognizing financial entities than general spaCy
```
- **Estimated Accuracy:** 82-88%
- **Speed:** 50-100 ms/document
- **Best Feature:** Trained on financial documents!

#### **C. RoBERTa / BERT-based NER**
```python
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")
text = "Transfer $5000 to DBS Bank"
entities = ner(text)
```
- **Estimated Accuracy:** 88-93%
- **Speed:** 80-150 ms/document

**Verdict:** ⭐⭐⭐⭐ **Highly recommended for Phase 2!**

---

### **4. Gensim** ⚠️ **Wrong Tool**

**What it does:** Topic modeling and document similarity

**Strengths:**
- ✅ Excellent for topic modeling (LDA, LSI)
- ✅ Good for document similarity
- ✅ Word2Vec, Doc2Vec implementations
- ✅ Fast and efficient

**Weaknesses:**
- ❌ **NOT for entity extraction/classification!**
- ❌ No NER capabilities
- ❌ No entity recognition features

**Best For:**
- Topic modeling (e.g., "What topics are in these 1000 documents?")
- Document clustering
- Semantic similarity
- **NOT for your use case!**

**Code Example:**
```python
from gensim.models import Word2Vec
# Used for word embeddings, NOT entity classification
```

**Verdict:** ❌ **Wrong tool for entity classification**

---

### **5. Flair** ⭐⭐⭐ **Worth Considering**

**What it does:** State-of-the-art NLP library with excellent NER

**Strengths:**
- ✅ Very accurate NER (often 90-95%)
- ✅ Multiple pre-trained models
- ✅ Easy to use
- ✅ Good for unknown entities

**Weaknesses:**
- ⚠️ Slower than spaCy (100-300ms)
- ⚠️ Less popular than Hugging Face

**Estimated Performance:**
- Entity Extraction: 92-96%
- Entity Classification: 85-90%
- Speed: 100-300 ms/document

**Code Example:**
```python
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')
sentence = Sentence("Transfer $5000 to DBS Bank")
tagger.predict(sentence)
for entity in sentence.get_spans('ner'):
    print(entity.text, entity.tag)
```

**Verdict:** ⭐⭐⭐ **Good alternative to Hugging Face**

---

### **6. Stanford NLP (Stanza)** ⭐⭐ **Accurate but Slow**

**What it does:** Stanford's NLP toolkit with neural models

**Strengths:**
- ✅ Very accurate
- ✅ Supports 60+ languages
- ✅ Academic backing (Stanford)

**Weaknesses:**
- ⚠️ Very slow (200-500ms)
- ⚠️ Large model size
- ⚠️ Complex setup

**Estimated Performance:**
- Entity Extraction: 90-94%
- Entity Classification: 82-88%
- Speed: 200-500 ms/document

**Verdict:** ⚠️ **Too slow for production**

---

### **7. Azure Text Analytics / AWS Comprehend / Google Cloud NLP** ☁️ **Cloud APIs**

**What they do:** Cloud-based NLP services

**Strengths:**
- ✅ Very accurate (85-92%)
- ✅ No setup needed
- ✅ Auto-scaling
- ✅ Finance-specific options

**Weaknesses:**
- ❌ **Cost per API call** (expensive at scale!)
- ❌ Requires internet
- ❌ Data privacy concerns (sending docs to cloud)
- ❌ Not suitable for on-prem banking systems

**Estimated Performance:**
- Entity Extraction: 90-95%
- Entity Classification: 85-92%
- Speed: 100-300 ms/document (network latency)
- **Cost:** $1-5 per 1000 documents

**Verdict:** ⚠️ **Good but expensive + privacy concerns**

---

## 📊 **Summary Comparison Table**

| Library | Accuracy | Speed | Handles Unknown | Finance-Aware | Cost | Recommendation |
|---------|----------|-------|----------------|---------------|------|----------------|
| **spaCy** | 76% ✅ | 4.6ms ⭐⭐⭐ | ❌ No | ❌ No | Free | ✅ Phase 1 baseline |
| **NLTK** | 40-50% ❌ | 50-100ms ⚠️ | ❌ No | ❌ No | Free | ❌ Skip |
| **Hugging Face (Zero-Shot)** | 85-90% ⭐⭐⭐ | 500-1000ms ⚠️ | ✅ YES! | ⚠️ Partial | Free | ⭐⭐⭐ RECOMMENDED Phase 2 |
| **Hugging Face (FinBERT)** | 82-88% ⭐⭐ | 50-100ms ✅ | ⚠️ Partial | ✅ YES | Free | ⭐⭐ RECOMMENDED Phase 2 |
| **Gensim** | N/A ❌ | N/A | ❌ No | ❌ No | Free | ❌ Wrong tool |
| **Flair** | 85-90% ⭐⭐⭐ | 100-300ms ⚠️ | ⚠️ Partial | ❌ No | Free | ⭐⭐ Alternative |
| **Stanza** | 82-88% ⭐⭐ | 200-500ms ❌ | ⚠️ Partial | ❌ No | Free | ⚠️ Too slow |
| **Cloud APIs** | 85-92% ⭐⭐⭐ | 100-300ms ⚠️ | ✅ YES | ✅ YES | $$$ | ⚠️ Privacy concerns |

---

## 🎯 **Recommendation for Your Project**

### **Phase 1: spaCy Baseline** ✅ **DONE**
- Fast and simple
- Good extraction (94.1%)
- Moderate classification (76.5%)
- **Status:** Completed

### **Phase 2: Hugging Face Transformers** ⭐⭐⭐ **IMPLEMENT NEXT**

Test 2-3 approaches:

1. **Zero-Shot Classification** (priority #1)
   - Handles unknown entities (Breadtalk, PropertyGuru)
   - Expected: 85-90% accuracy on unknowns
   - Model: `facebook/bart-large-mnli`

2. **FinBERT** (priority #2)
   - Finance-specific
   - Expected: 82-88% overall accuracy
   - Model: `ProsusAI/finbert`

3. **BERT-based NER** (optional)
   - General purpose
   - Expected: 88-93% accuracy
   - Model: `dslim/bert-base-NER`

### **Phase 3: Hybrid System** (production)
- spaCy for fast extraction (94.1%)
- Zero-Shot for unknown entity classification
- Small KB for top 100 frequent entities
- **Expected:** 88-92% overall, 50-100ms speed

---

## 💡 **Libraries to SKIP for Your Use Case**

❌ **NLTK** - Too basic, poor accuracy  
❌ **Gensim** - Not for entity classification  
❌ **Stanza** - Too slow for production  

---

## 🚀 **Next Steps**

1. ✅ **Expand dataset** (DONE - 51 → 81 examples)
2. ⏭️ **Implement Zero-Shot Classification** (handles unknown entities)
3. ⏭️ **Test FinBERT** (finance-specific)
4. ⏭️ **Create comparison report** (spaCy vs Zero-Shot vs FinBERT)
5. ⏭️ **Build hybrid system** (best of all approaches)

---

**Bottom Line:** Use **Hugging Face Transformers** with **Zero-Shot Classification** for Phase 2 to handle unknown entities!
