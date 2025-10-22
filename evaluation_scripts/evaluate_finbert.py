"""
FinBERT Evaluation
Finance-specific NLP model for entity recognition.
Trained on financial documents for better accuracy in banking context.
"""

import sys
import time
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
except ImportError:
    print("❌ Missing 'transformers'. Install with: pip install transformers torch")
    sys.exit(1)

try:
    import spacy
except ImportError:
    print("❌ Missing 'spacy'. Install with: python -m pip install spacy")
    sys.exit(1)

from utils_nlp import (
    load_ground_truth_dataset,
    print_dataset_stats,
    print_results_by_category,
    print_results_by_familiarity,
    calculate_overall_accuracy,
    save_results_to_json
)


def extract_entities_finbert(text, ner_pipeline):
    """Extract entities using FinBERT."""
    try:
        entities = ner_pipeline(text)
        # Filter and format results
        formatted_entities = []
        for ent in entities:
            formatted_entities.append({
                'text': ent.get('word', ''),
                'label': ent.get('entity', ''),
                'score': ent.get('score', 0.0)
            })
        return formatted_entities
    except Exception as e:
        print(f"⚠️  Error extracting entities: {e}")
        return []


def evaluate_finbert(dataset):
    """Evaluate FinBERT for finance-specific NER."""
    print("\n" + "="*70)
    print("FINBERT EVALUATION (Finance-Specific NER)")
    print("="*70)
    print("⭐ Model: ProsusAI/finbert")
    print("⭐ Trained on financial documents!")
    print("="*70)
    
    # Load FinBERT model
    print("\n🔧 Loading FinBERT model...")
    print("   (This may take a minute on first load...)")
    
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=-1)
        print("✅ FinBERT model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading FinBERT: {e}")
        print("\n💡 Note: FinBERT may not have a pre-trained NER model.")
        print("   Alternative finance models:")
        print("   - xlm-roberta-large-finetuned-conll03-english")
        print("   - dslim/bert-base-NER-uncased")
        print("   - Jean-Baptiste/roberta-large-ner-english")
        return None
    
    results_by_category = defaultdict(lambda: {'total': 0, 'correct': 0})
    results_by_known = {'known': {'total': 0, 'correct': 0}, 'unknown': {'total': 0, 'correct': 0}}
    
    total_time = 0
    processed = 0
    
    print("\n⏳ Processing entities...")
    
    for i, item in enumerate(dataset):
        text = item['text']
        ground_truth_entity = item['entity_text']
        ground_truth_label = item['entity_label']
        ground_truth_category = item['entity_category']
        is_well_known = item['is_well_known']
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{len(dataset)} entities...")
        
        # Extract entities with FinBERT
        start_time = time.time()
        entities = extract_entities_finbert(text, ner_pipeline)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        processed += 1
        
        # Find matching entity
        matched_entity = None
        for e in entities:
            if ground_truth_entity in e['text'] or e['text'] in ground_truth_entity:
                matched_entity = e
                break
        
        # Track stats
        results_by_category[ground_truth_category]['total'] += 1
        known_key = 'known' if is_well_known else 'unknown'
        results_by_known[known_key]['total'] += 1
        
        # Check if correct
        if matched_entity and matched_entity['label'] == ground_truth_label:
            results_by_category[ground_truth_category]['correct'] += 1
            results_by_known[known_key]['correct'] += 1
    
    print(f"\n✅ Processed {processed} entities")
    
    # Print results
    print_results_by_category(results_by_category)
    print_results_by_familiarity(results_by_known)
    
    # Overall accuracy
    overall_accuracy = calculate_overall_accuracy(results_by_category)
    avg_time = (total_time / processed) * 1000
    
    print("\n" + "="*70)
    print(f"🎯 OVERALL ACCURACY: {overall_accuracy:.1f}%")
    print(f"⏱️  AVERAGE TIME: {avg_time:.1f} ms/entity")
    print(f"🚀  THROUGHPUT: {processed/total_time:.1f} entities/second")
    print("="*70)
    
    return {
        'model': 'ProsusAI/finbert',
        'approach': 'FinBERT',
        'overall_accuracy': overall_accuracy,
        'avg_time_ms': avg_time,
        'throughput': processed / total_time,
        'processed': processed,
        'by_category': dict(results_by_category),
        'by_known': results_by_known
    }


def main():
    print("="*70)
    print("FINBERT EVALUATION (Finance-Specific NER)")
    print("="*70)
    print("Approach: Finance-specific BERT model")
    print("Model: ProsusAI/finbert")
    print("="*70)
    
    # Load dataset
    dataset_path = "banking_ner_dataset.csv"
    print(f"\n📂 Loading dataset: {dataset_path}")
    
    try:
        dataset = load_ground_truth_dataset(dataset_path)
        print_dataset_stats(dataset)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {dataset_path}")
        sys.exit(1)
    
    # Evaluate FinBERT
    results = evaluate_finbert(dataset)
    
    if results:
        # Save results
        save_results_to_json(results, 'results_finbert.json')
        
        print("\n" + "="*70)
        print("✅ FinBERT evaluation complete!")
        print("="*70)
        print("\n💡 Key Advantages:")
        print("   ✅ Finance-specific training")
        print("   ✅ Better understanding of banking terminology")
        print("   ✅ Optimized for financial documents")
        print("\n⚠️  Trade-offs:")
        print("   ⚠️  May not cover all entity types")
        print("   ⚠️  Slower than spaCy")
        print("="*70)


if __name__ == "__main__":
    main()
