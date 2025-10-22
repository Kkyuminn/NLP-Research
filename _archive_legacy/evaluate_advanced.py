"""
Phase 2: Advanced NLP Approaches Evaluation
Test Zero-Shot Classification and FinBERT for entity classification.
"""

import sys
import csv
import time
from collections import defaultdict

try:
    import spacy
except ImportError:
    print("Missing 'spacy'. Install with: python -m pip install spacy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Missing 'pandas'. Install with: python -m pip install pandas")
    sys.exit(1)


def load_ground_truth_dataset(csv_path):
    """Load the labeled dataset with ground truth."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'text': row['text'],
                'entity_text': row['entity_text'],
                'entity_label': row['entity_label'],
                'entity_category': row['entity_category'],
                'is_well_known': row['is_well_known'] == 'TRUE'
            })
    return data


def extract_entities_spacy(text, nlp):
    """Extract entities using spaCy."""
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


def classify_entity_zero_shot(entity_text, candidate_labels):
    """
    Classify entity using Zero-Shot Classification.
    This approach can classify entities it has never seen before!
    """
    try:
        from transformers import pipeline
        
        # Check if classifier is already loaded
        if not hasattr(classify_entity_zero_shot, 'classifier'):
            print("🔧 Loading Zero-Shot Classification model (first time only)...")
            classify_entity_zero_shot.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU
            )
        
        classifier = classify_entity_zero_shot.classifier
        
        # Classify the entity
        result = classifier(entity_text, candidate_labels=candidate_labels)
        
        # Return top prediction with confidence score
        return {
            'label': result['labels'][0],
            'confidence': result['scores'][0]
        }
    
    except ImportError:
        print("\n⚠️  WARNING: transformers not installed")
        print("   Install with: pip install transformers torch")
        return None
    except Exception as e:
        print(f"\n⚠️  Error in Zero-Shot Classification: {e}")
        return None


def evaluate_zero_shot(dataset, nlp):
    """Evaluate Zero-Shot Classification approach."""
    print("\n" + "="*70)
    print("ZERO-SHOT CLASSIFICATION EVALUATION")
    print("="*70)
    print("⭐ This approach can classify entities it has NEVER seen before!")
    
    # Define candidate labels based on entity categories in dataset
    candidate_labels = [
        "bank", "hospital", "university", "school", "government agency",
        "tech company", "e-commerce company", "retail company", "restaurant",
        "food company", "insurance company", "telecommunications company",
        "transportation company", "real estate company", "pharmaceutical company",
        "construction company", "social media company", "entertainment company",
        "person", "customer", "employee", "executive",
        "money amount", "currency", "date", "time",
        "country", "city", "location", "street", "building"
    ]
    
    # Map our entity categories to candidate labels
    category_to_label = {
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
        'Security Services Company': 'tech company',
        'Industrial Company': 'construction company',
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
    
    results_by_category = defaultdict(lambda: {'total': 0, 'correct': 0})
    results_by_known = {'known': {'total': 0, 'correct': 0}, 'unknown': {'total': 0, 'correct': 0}}
    
    total_time = 0
    processed = 0
    
    print("\n⏳ Processing entities (this may take a while)...")
    
    for i, item in enumerate(dataset):
        ground_truth_category = item['entity_category']
        entity_text = item['entity_text']
        is_well_known = item['is_well_known']
        
        # Only classify ORG and PERSON entities (others are simple)
        if item['entity_label'] not in ['ORG', 'PERSON']:
            continue
        
        if i % 10 == 0:
            print(f"   Processed {i}/{len(dataset)} entities...")
        
        # Classify using Zero-Shot
        start_time = time.time()
        result = classify_entity_zero_shot(entity_text, candidate_labels)
        end_time = time.time()
        
        if result is None:
            print("\n❌ Zero-Shot Classification not available. Skipping evaluation.")
            return None
        
        total_time += (end_time - start_time)
        processed += 1
        
        # Check if prediction matches ground truth
        predicted_label = result['label']
        expected_label = category_to_label.get(ground_truth_category, ground_truth_category.lower())
        
        results_by_category[ground_truth_category]['total'] += 1
        known_key = 'known' if is_well_known else 'unknown'
        results_by_known[known_key]['total'] += 1
        
        # Consider it correct if labels match closely
        if predicted_label == expected_label or predicted_label in expected_label or expected_label in predicted_label:
            results_by_category[ground_truth_category]['correct'] += 1
            results_by_known[known_key]['correct'] += 1
    
    if processed == 0:
        return None
    
    # Print results
    print("\n📊 Accuracy by Entity Category:")
    print(f"{'Category':<35} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 70)
    
    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{category:<35} {stats['total']:<8} {stats['correct']:<8} {accuracy:.1f}%")
    
    print("\n📊 Accuracy by Entity Familiarity:")
    print(f"{'Type':<35} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 70)
    
    for known_type in ['known', 'unknown']:
        stats = results_by_known[known_type]
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        label = "Well-known entities" if known_type == 'known' else "Unknown/Local entities"
        print(f"{label:<35} {stats['total']:<8} {stats['correct']:<8} {accuracy:.1f}%")
    
    # Overall accuracy
    total_correct = sum(cat['correct'] for cat in results_by_category.values())
    total_items = sum(cat['total'] for cat in results_by_category.values())
    overall_accuracy = (total_correct / total_items * 100) if total_items > 0 else 0
    
    avg_time = (total_time / processed) * 1000
    
    print("\n" + "="*70)
    print(f"🎯 OVERALL ACCURACY: {overall_accuracy:.1f}%")
    print(f"⏱️  AVERAGE TIME: {avg_time:.1f} ms/entity")
    print("="*70)
    
    return {
        'overall_accuracy': overall_accuracy,
        'avg_time_ms': avg_time,
        'by_category': dict(results_by_category),
        'by_known': results_by_known
    }


def evaluate_finbert(dataset, nlp):
    """Evaluate FinBERT for finance-specific NER."""
    print("\n" + "="*70)
    print("FINBERT EVALUATION (Finance-Specific Model)")
    print("="*70)
    print("⭐ This model is trained on financial documents!")
    
    print("\n⚠️  FinBERT evaluation requires specialized setup.")
    print("   This would require: pip install transformers torch")
    print("   Skipping for now - to be implemented in future phase.")
    
    return None


def compare_approaches(spacy_results, zero_shot_results):
    """Create comparison table of different approaches."""
    print("\n" + "="*70)
    print("📊 APPROACH COMPARISON TABLE")
    print("="*70)
    
    print(f"\n{'Approach':<20} {'Accuracy':<15} {'Speed':<20} {'Unknown Entities':<20}")
    print("-" * 80)
    
    # spaCy baseline
    if spacy_results:
        print(f"{'spaCy (Baseline)':<20} {spacy_results.get('overall_accuracy', 0):.1f}%{'':<10} "
              f"{spacy_results.get('avg_time_ms', 0):.1f} ms{'':<10} "
              f"{spacy_results.get('by_known', {}).get('unknown', {}).get('correct', 0)}"
              f"/{spacy_results.get('by_known', {}).get('unknown', {}).get('total', 0)}")
    
    # Zero-Shot
    if zero_shot_results:
        print(f"{'Zero-Shot':<20} {zero_shot_results.get('overall_accuracy', 0):.1f}%{'':<10} "
              f"{zero_shot_results.get('avg_time_ms', 0):.1f} ms{'':<10} "
              f"{zero_shot_results.get('by_known', {}).get('unknown', {}).get('correct', 0)}"
              f"/{zero_shot_results.get('by_known', {}).get('unknown', {}).get('total', 0)}")


def main():
    print("="*70)
    print("PHASE 2: ADVANCED NLP APPROACHES EVALUATION")
    print("="*70)
    print("Testing: Zero-Shot Classification + FinBERT")
    print("="*70)
    
    # Load dataset
    dataset_path = "banking_ner_dataset.csv"
    print(f"\n📂 Loading dataset: {dataset_path}")
    
    try:
        dataset = load_ground_truth_dataset(dataset_path)
        print(f"✅ Loaded {len(dataset)} labeled examples")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {dataset_path}")
        sys.exit(1)
    
    # Load spaCy for entity extraction
    print("\n🔧 Loading spaCy model for entity extraction...")
    try:
        nlp = spacy.load('en_core_web_md')
        print("✅ Model loaded successfully")
    except OSError:
        print("❌ Error: Install with: python -m spacy download en_core_web_md")
        sys.exit(1)
    
    # Evaluate approaches
    print("\n" + "="*70)
    print("🚀 STARTING EVALUATIONS")
    print("="*70)
    
    # 1. Zero-Shot Classification
    zero_shot_results = evaluate_zero_shot(dataset, nlp)
    
    # 2. FinBERT (placeholder for now)
    finbert_results = evaluate_finbert(dataset, nlp)
    
    # Compare results
    if zero_shot_results:
        compare_approaches(None, zero_shot_results)
    
    print("\n" + "="*70)
    print("✅ Phase 2 evaluation complete!")
    print("="*70)
    
    print("\n💡 Next Steps:")
    print("   1. Fine-tune candidate labels for better accuracy")
    print("   2. Implement hybrid approach (spaCy + Zero-Shot)")
    print("   3. Build production pipeline with caching")
    print("="*70)


if __name__ == "__main__":
    main()
