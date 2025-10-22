"""
Zero-Shot Classification Evaluation
Uses Hugging Face Transformers for entity classification without training.
Can classify entities it has NEVER seen before!
"""

import sys
import time
from collections import defaultdict

try:
    from transformers import pipeline
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


# Entity category mapping
CATEGORY_TO_LABEL = {
    'Bank': 'bank',
    'Hospital': 'hospital',
    'University': 'university',
    'Educational Institution': 'school',
    'Government Agency': 'government agency',
    'Tech Company': 'tech company',
    'E-commerce Company': 'e-commerce company',
    'Retail Company': 'retail company',
    'Food & Beverage Company': 'restaurant',
    'Food Manufacturing Company': 'food company',
    'Insurance Company': 'insurance company',
    'Telecommunications Company': 'telecommunications company',
    'Transportation Company': 'transportation company',
    'Real Estate Company': 'real estate company',
    'Real Estate Investment Trust': 'real estate company',
    'Real Estate Tech Company': 'tech company',
    'Pharmaceutical Company': 'pharmaceutical company',
    'Construction Company': 'construction company',
    'Social Media Company': 'social media company',
    'Entertainment Company': 'entertainment company',
    'Security Services Company': 'security company',
    'Industrial Company': 'industrial company',
    'Customer': 'customer',
    'CEO': 'executive',
    'Employee': 'employee',
    'Executive': 'executive',
    'Amount': 'money amount',
    'Transaction Date': 'date',
    'Period Start': 'date',
    'Period End': 'date',
    'Maturity Date': 'date',
    'Recurring Date': 'date',
    'Country': 'country',
    'Street': 'street',
    'Location': 'location',
    'Facility': 'building'
}

# Candidate labels for classification
CANDIDATE_LABELS = [
    "bank", "hospital", "university", "school", "government agency",
    "tech company", "e-commerce company", "retail company", "restaurant",
    "food company", "insurance company", "telecommunications company",
    "transportation company", "real estate company", "pharmaceutical company",
    "construction company", "industrial company", "security company",
    "social media company", "entertainment company",
    "person", "customer", "employee", "executive", "CEO",
    "money amount", "currency", "date", "time",
    "country", "city", "location", "street", "building"
]


def extract_entities_spacy(text, nlp):
    """Extract entities using spaCy for initial detection."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    return entities


def classify_entity_zero_shot(entity_text, classifier, candidate_labels):
    """Classify entity using Zero-Shot Classification."""
    try:
        result = classifier(entity_text, candidate_labels=candidate_labels)
        return {
            'label': result['labels'][0],
            'confidence': result['scores'][0]
        }
    except Exception as e:
        print(f"⚠️  Error classifying '{entity_text}': {e}")
        return None


def evaluate_zero_shot(dataset, nlp):
    """Evaluate Zero-Shot Classification approach."""
    print("\n" + "="*70)
    print("ZERO-SHOT CLASSIFICATION EVALUATION")
    print("="*70)
    print("⭐ Model: facebook/bart-large-mnli")
    print("⭐ Can classify entities NEVER seen in training data!")
    print("="*70)
    
    # Load Zero-Shot classifier
    print("\n🔧 Loading Zero-Shot Classification model...")
    print("   (This may take a minute on first load...)")
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # Use CPU
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    results_by_category = defaultdict(lambda: {'total': 0, 'correct': 0})
    results_by_known = {'known': {'total': 0, 'correct': 0}, 'unknown': {'total': 0, 'correct': 0}}
    
    total_time = 0
    processed = 0
    
    print("\n⏳ Processing entities...")
    print("   (Only classifying ORG and PERSON entities)")
    
    for i, item in enumerate(dataset):
        ground_truth_category = item['entity_category']
        entity_text = item['entity_text']
        is_well_known = item['is_well_known']
        entity_label = item['entity_label']
        
        # Only classify ORG and PERSON entities (others like MONEY, DATE are simple)
        if entity_label not in ['ORG', 'PERSON']:
            continue
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{len(dataset)} entities...")
        
        # Classify using Zero-Shot
        start_time = time.time()
        result = classify_entity_zero_shot(entity_text, classifier, CANDIDATE_LABELS)
        end_time = time.time()
        
        if result is None:
            continue
        
        total_time += (end_time - start_time)
        processed += 1
        
        # Check if prediction matches ground truth
        predicted_label = result['label']
        expected_label = CATEGORY_TO_LABEL.get(ground_truth_category, ground_truth_category.lower())
        
        results_by_category[ground_truth_category]['total'] += 1
        known_key = 'known' if is_well_known else 'unknown'
        results_by_known[known_key]['total'] += 1
        
        # Check if prediction is correct (exact or partial match)
        is_correct = (
            predicted_label == expected_label or 
            predicted_label in expected_label or 
            expected_label in predicted_label
        )
        
        if is_correct:
            results_by_category[ground_truth_category]['correct'] += 1
            results_by_known[known_key]['correct'] += 1
    
    if processed == 0:
        print("\n❌ No entities were processed!")
        return None
    
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
        'model': 'facebook/bart-large-mnli',
        'approach': 'Zero-Shot Classification',
        'overall_accuracy': overall_accuracy,
        'avg_time_ms': avg_time,
        'throughput': processed / total_time,
        'processed': processed,
        'by_category': dict(results_by_category),
        'by_known': results_by_known
    }


def main():
    print("="*70)
    print("ZERO-SHOT CLASSIFICATION EVALUATION")
    print("="*70)
    print("Approach: Hugging Face Transformers")
    print("Model: facebook/bart-large-mnli")
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
    
    # Load spaCy for entity extraction
    print("\n🔧 Loading spaCy for entity extraction...")
    try:
        nlp = spacy.load('en_core_web_md')
        print("✅ spaCy model loaded")
    except OSError:
        print("❌ Error: Install with: python -m spacy download en_core_web_md")
        sys.exit(1)
    
    # Evaluate Zero-Shot
    results = evaluate_zero_shot(dataset, nlp)
    
    if results:
        # Save results
        save_results_to_json(results, 'results_zero_shot.json')
        
        print("\n" + "="*70)
        print("✅ Zero-Shot evaluation complete!")
        print("="*70)
        print("\n💡 Key Advantages:")
        print("   ✅ Can classify unknown entities (never seen in training)")
        print("   ✅ No training required")
        print("   ✅ Works with any entity category")
        print("\n⚠️  Trade-offs:")
        print("   ⚠️  Slower than spaCy (500-1000ms vs 4ms)")
        print("   ⚠️  Requires more memory")
        print("="*70)


if __name__ == "__main__":
    main()
