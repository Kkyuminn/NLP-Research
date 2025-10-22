import sys
try:
    import datasets
    from datasets import load_dataset
except ImportError:
    print("Missing 'datasets'. Install in this interpreter with: python -m pip install datasets")
    print(f"Interpreter: {sys.executable}")
    raise
try:
    import spacy
except ImportError:
    print("Missing 'spacy'. Install with: python -m pip install spacy && python -m spacy download en_core_web_md")
    print(f"Interpreter: {sys.executable}")
    raise


def load_sts_dataset():
    """Load the STS Benchmark dataset using Hugging Face Datasets."""
    dataset = load_dataset('stsb_multi_mt', name='en')
    return dataset


def get_spacy_similarity(sentence1, sentence2, nlp):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    return doc1.similarity(doc2)


def extract_entities_and_pos(text, nlp):
    """Extract named entities and POS tags from text."""
    doc = nlp(text)
    
    # Named Entity Recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Part-of-Speech tagging
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    return entities, pos_tags


def evaluate_ner_accuracy(predictions, ground_truth):
    """
    Calculate precision, recall, and F1-score for NER.
    
    predictions: list of (text, label) tuples from model
    ground_truth: list of (text, label) tuples (correct answers)
    """
    # Convert to sets for comparison (exact match)
    pred_set = set(predictions)
    true_set = set(ground_truth)
    
    # True Positives: correctly predicted entities
    true_positives = len(pred_set & true_set)
    
    # False Positives: predicted but not in ground truth
    false_positives = len(pred_set - true_set)
    
    # False Negatives: in ground truth but not predicted
    false_negatives = len(true_set - pred_set)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


if __name__ == "__main__":
    print(f"Interpreter: {sys.executable}")
    print(f"datasets version: {datasets.__version__}, spacy version: {spacy.__version__}\n")
    
    nlp = spacy.load('en_core_web_md')
    
    # Example sentence
    text = "Hi Bob, I am at the Lepark Korner"
    
    # Extract entities and POS tags
    entities, pos_tags = extract_entities_and_pos(text, nlp)
    
    print(f"Text: {text}\n")
    print("Named Entities (NER):")
    for entity_text, entity_label in entities:
        print(f"  - '{entity_text}' → {entity_label}")
    
    print("\nPart-of-Speech Tags:")
    for word, pos in pos_tags:
        print(f"  - '{word}' → {pos}")
    
    # --- ACCURACY EVALUATION EXAMPLE ---
    print("\n" + "="*60)
    print("ACCURACY EVALUATION EXAMPLE")
    print("="*60)
    
    # Ground truth (what the entities SHOULD be)
    ground_truth = [
        ("Bob", "PERSON"),
        ("Lepark Korner", "GPE")  # GPE = Geopolitical Entity (location)
    ]
    
    # Model predictions
    predictions = entities
    
    print(f"\nGround Truth: {ground_truth}")
    print(f"Model Predictions: {predictions}")
    
    # Calculate accuracy metrics
    metrics = evaluate_ner_accuracy(predictions, ground_truth)
    
    print(f"\n📊 Accuracy Metrics:")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1_score']:.2%}")
    print(f"\n  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # --- SIMILARITY EXAMPLE (original code) ---
    print("\n" + "="*60)
    print("SIMILARITY EXAMPLE")
    print("="*60)
    s1 = "Both the mathematics equations are fraudulent"
    s2 = "Both the mathematics equations are fraudulent"
    score = get_spacy_similarity(s1, s2, nlp)
    print(f"\nSentence 1: {s1}")
    print(f"Sentence 2: {s2}")
    print(f"spaCy similarity score: {score}")