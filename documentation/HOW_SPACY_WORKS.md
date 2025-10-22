# How spaCy Evaluation Works - Step by Step

## 🎯 Overview
The spaCy evaluation script tests how well spaCy's pre-trained model can **extract** and **classify** entities from banking documents, then compares the results against manually labeled "ground truth" data.

---

## 📊 The Big Picture

```
Input: Banking text → spaCy Model → Output: Entities → Compare → Accuracy Score
                                                           ↓
                                                    Ground Truth
                                                    (Your labels)
```

---

## 🔍 Step-by-Step Process

### **STEP 1: Load the Ground Truth Dataset** 📂

**What happens:**
```python
dataset = load_ground_truth_dataset("banking_ner_dataset.csv")
```

**Example CSV data:**
```csv
text,entity_text,entity_label,entity_category,is_well_known
"Transfer $5000 from John Doe to DBS Bank","DBS Bank","ORG","Bank",TRUE
"Transfer $5000 from John Doe to DBS Bank","John Doe","PERSON","Customer",TRUE
"Transfer $5000 from John Doe to DBS Bank","5000","MONEY","Amount",TRUE
```

**Result:**
- Loads 81 labeled examples
- Each row contains:
  - `text` = The full banking document text
  - `entity_text` = What entity we expect to find (e.g., "DBS Bank")
  - `entity_label` = spaCy's entity type (e.g., "ORG", "PERSON", "MONEY")
  - `entity_category` = Our custom category (e.g., "Bank", "Customer", "Amount")
  - `is_well_known` = Whether the entity is famous (TRUE/FALSE)

---

### **STEP 2: Load spaCy Model** 🤖

**What happens:**
```python
nlp = spacy.load('en_core_web_md')
```

**This loads a pre-trained neural network model that:**
- Was trained on millions of English documents
- Learned patterns to recognize entities (names, organizations, dates, money, etc.)
- Can process text and identify 18 entity types:
  - **PERSON** - People's names
  - **ORG** - Organizations, companies, agencies
  - **MONEY** - Monetary values
  - **DATE** - Dates
  - **GPE** - Countries, cities, states
  - **LOC** - Locations (non-GPE)
  - **FAC** - Facilities, buildings
  - And 11 more types...

**Model size:** ~50MB (medium model)

---

### **STEP 3: Entity Extraction Evaluation** ✅❌

**What happens:**
```python
extraction = evaluate_entity_extraction(dataset, nlp)
```

#### **3.1 Process Each Text**
For each row in the dataset:

**Example:**
```
Text: "Transfer $5000 from John Doe to account 123-456-789 at DBS Bank"
Expected entity: "DBS Bank"
```

#### **3.2 Run spaCy on the Text**
```python
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
```

**spaCy's neural network:**
1. Tokenizes the text (breaks into words)
2. Analyzes each word's context
3. Predicts entity boundaries and types
4. Returns found entities

**spaCy finds:**
```python
[
  ("$5000", "MONEY"),
  ("John Doe", "PERSON"),
  ("DBS Bank", "ORG")
]
```

#### **3.3 Check if Expected Entity Was Found**
```python
if "DBS Bank" in found_entities:
    found += 1  # ✅ Success!
else:
    not_found.append("DBS Bank")  # ❌ Missed
```

#### **3.4 Calculate Extraction Accuracy**
```python
accuracy = (found / total) * 100
# Result: 93.8% (76 out of 81 entities found)
```

**Interpretation:**
- ✅ **93.8%** - spaCy successfully detected the entity in the text
- ❌ **6.2%** - spaCy missed the entity (e.g., "Twitter", "Grab Holdings")

---

### **STEP 4: Entity Classification Evaluation** 🏷️

**What happens:**
```python
classification = evaluate_entity_classification(dataset, nlp)
```

This checks if spaCy **correctly labeled** the entity type.

#### **4.1 For Each Entity Found**

**Example 1: ✅ Correct**
```
Text: "Transfer to DBS Bank"
Expected: entity="DBS Bank", label="ORG"
spaCy found: entity="DBS Bank", label="ORG"
Result: ✅ CORRECT! (matches)
```

**Example 2: ❌ Wrong**
```
Text: "Payment of $5000"
Expected: entity="5000", label="MONEY"
spaCy found: entity="5000", label="CARDINAL"
Result: ❌ WRONG! (spaCy used wrong label)
```

**Example 3: ❌ Not Found**
```
Text: "Payment from Twitter"
Expected: entity="Twitter", label="ORG"
spaCy found: [] (nothing)
Result: ❌ WRONG! (couldn't find entity at all)
```

#### **4.2 Track Results by Category**
```python
results_by_category = {
    'Bank': {'total': 4, 'correct': 4},      # 100% accuracy
    'Hospital': {'total': 4, 'correct': 4},  # 100% accuracy
    'Amount': {'total': 7, 'correct': 2},    # 28.6% accuracy
    'E-commerce': {'total': 2, 'correct': 0} # 0% accuracy
}
```

#### **4.3 Track by Known vs Unknown Entities**
```python
results_by_known = {
    'known': {'total': 66, 'correct': 44},    # 66.7% (well-known like "DBS", "NUS")
    'unknown': {'total': 15, 'correct': 11}   # 73.3% (local like "Breadtalk", "Koufu")
}
```

#### **4.4 Calculate Overall Classification Accuracy**
```python
overall_accuracy = (55 / 81) * 100 = 67.9%
```

**Interpretation:**
- Only **67.9%** of entities were correctly classified
- spaCy is good at detecting (93.8%) but weaker at classifying (67.9%)

---

### **STEP 5: Speed Benchmark** ⚡

**What happens:**
```python
speed = benchmark_speed(dataset, nlp, sample_size=50)
```

#### **5.1 Process 50 Documents**
```python
start_time = time.time()
for text in texts:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
end_time = time.time()
```

#### **5.2 Calculate Metrics**
```python
total_time = 0.21 seconds (for 50 documents)
avg_time = 4.2 ms per document
throughput = 237 documents/second
```

**Interpretation:**
- ⚡ **Very fast!** - Can process 237 documents per second
- ✅ **Production-ready** - 4.2ms is fast enough for real-time systems

---

### **STEP 6: Print Summary** 📋

**Final output:**
```
Entity Extraction Accuracy:     93.8%
Entity Classification Accuracy: 67.9%
Processing Speed:               4.2 ms/document
Throughput:                     237 docs/second
```

---

## 📊 Detailed Example Walkthrough

### **Example 1: Successful Detection and Classification**

**Input:**
```
Text: "Wire transfer to OCBC Bank account for property purchase"
Expected: entity="OCBC Bank", label="ORG", category="Bank"
```

**Step 1:** spaCy processes text
```python
doc = nlp("Wire transfer to OCBC Bank account for property purchase")
```

**Step 2:** spaCy finds entities
```python
entities = [("OCBC Bank", "ORG")]  # ✅ Found!
```

**Step 3:** Check extraction
```python
"OCBC Bank" in ["OCBC Bank"]  # ✅ True - Entity extracted!
extraction_accuracy += 1
```

**Step 4:** Check classification
```python
Expected label: "ORG"
spaCy's label:  "ORG"
"ORG" == "ORG"  # ✅ True - Correctly classified!
classification_accuracy += 1
```

**Result:** ✅✅ **Full success!**

---

### **Example 2: Found but Wrongly Classified**

**Input:**
```
Text: "Payment of $2500 to vendor"
Expected: entity="2500", label="MONEY", category="Amount"
```

**Step 1:** spaCy processes text
```python
doc = nlp("Payment of $2500 to vendor")
```

**Step 2:** spaCy finds entities
```python
entities = [("$2500", "MONEY")]  # ✅ Found!
```

**Step 3:** Check extraction
```python
"2500" in "$2500"  # ✅ True (partial match) - Entity extracted!
extraction_accuracy += 1
```

**Step 4:** Check classification
```python
Expected label: "MONEY"
spaCy's label:  "MONEY"
"MONEY" == "MONEY"  # ✅ True - Correctly classified!
classification_accuracy += 1
```

**Result:** ✅✅ **Full success!**

---

### **Example 3: Completely Missed**

**Input:**
```
Text: "Payment received from Twitter for advertising services"
Expected: entity="Twitter", label="ORG", category="Social Media Company"
```

**Step 1:** spaCy processes text
```python
doc = nlp("Payment received from Twitter for advertising services")
```

**Step 2:** spaCy finds entities
```python
entities = []  # ❌ Nothing found!
```

**Step 3:** Check extraction
```python
"Twitter" in []  # ❌ False - Entity NOT extracted!
not_found.append("Twitter")
```

**Step 4:** Check classification
```python
# Can't check classification because entity wasn't found
classification_accuracy += 0
```

**Result:** ❌❌ **Complete failure** - spaCy didn't recognize "Twitter"

---

## 🧠 How spaCy's Neural Network Works (Simplified)

### **1. Training Phase** (Already done by spaCy)
```
Millions of documents → Neural Network → Learns patterns
"The CEO of Apple announced..." → Apple = ORG
"John Smith works at..." → John Smith = PERSON
"Payment of $500..." → $500 = MONEY
```

### **2. Prediction Phase** (What happens when you run the script)
```
Your text: "Transfer to DBS Bank"
           ↓
Neural Network analyzes each word:
- "Transfer" → not an entity
- "to" → not an entity
- "DBS" → looks like organization start
- "Bank" → looks like organization continuation
           ↓
Output: "DBS Bank" = ORG (confidence: 95%)
```

### **3. Why It Makes Mistakes**
❌ **"Twitter" not recognized:**
- Training data might not have many social media company examples
- "Twitter" might appear in different contexts
- Model learned patterns for banks/hospitals (common in financial docs)

❌ **Money amounts (28.6% accuracy):**
- spaCy sometimes labels "$5000" as CARDINAL instead of MONEY
- Depends on context and formatting

---

## 📊 Accuracy Metrics Explained

### **Extraction Accuracy = 93.8%**
"Did spaCy FIND the entity in the text?"
- ✅ Found: 76 out of 81 entities
- ❌ Missed: 5 entities (Twitter, Grab Holdings, ComfortDelGro, etc.)

### **Classification Accuracy = 67.9%**
"Did spaCy label the entity with the CORRECT type?"
- ✅ Correct: 55 out of 81 entities
- ❌ Wrong: 26 entities (wrong label or not found)

### **Why Classification < Extraction?**
Even if spaCy finds an entity, it might:
- Use the wrong label (CARDINAL instead of MONEY)
- Not recognize the entity type at all

---

## 🎯 Key Takeaways

1. **spaCy is FAST** ⚡ - 4.2ms per document (237 docs/second)
2. **Good at finding entities** ✅ - 93.8% extraction accuracy
3. **Weaker at classifying them** ⚠️ - 67.9% classification accuracy
4. **Struggles with:**
   - Unknown entities (Twitter, Grab)
   - E-commerce companies (0% accuracy)
   - Money amounts (28.6% accuracy)
   - Dates (50% accuracy)
5. **Excels at:**
   - Banks (100% accuracy)
   - Hospitals (100% accuracy)
   - Universities (100% accuracy)

---

## 🔄 The Evaluation Loop (Visual)

```
For each of 81 examples:
  ┌─────────────────────────────────────┐
  │ 1. Get text + expected entity       │
  │    "Transfer to DBS Bank" / "DBS Bank"
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 2. Run spaCy on text                │
  │    nlp("Transfer to DBS Bank")      │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 3. Check if entity found            │
  │    "DBS Bank" in results? ✅        │
  │    extraction_score += 1            │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 4. Check if label correct           │
  │    "ORG" == "ORG"? ✅              │
  │    classification_score += 1        │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 5. Track by category & known/unknown│
  │    Bank: 4/4 ✅                     │
  │    Known: 44/66                     │
  └─────────────────────────────────────┘

Final scores:
  Extraction: 76/81 = 93.8%
  Classification: 55/81 = 67.9%
```

---

## 💡 Why We Need This Evaluation

Without evaluation, we don't know:
- ❓ How accurate is spaCy for banking documents?
- ❓ What types of entities does it struggle with?
- ❓ Is it fast enough for production?
- ❓ Should we use a different NLP approach?

**With evaluation, we can:**
- ✅ Measure accuracy objectively (67.9%)
- ✅ Identify weaknesses (e-commerce, money, dates)
- ✅ Confirm speed is acceptable (4.2ms)
- ✅ Compare with other approaches (Zero-Shot, FinBERT)
- ✅ Make data-driven decisions for production

---

**🎉 Now you understand how the spaCy evaluation works step by step!**
