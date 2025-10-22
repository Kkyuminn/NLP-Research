"""
Phase 1: Entity Classification Evaluation - spaCy Baseline
Compare spaCy's performance on entity extraction and classification.
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


def evaluate_entity_extraction(dataset, nlp):
    """Evaluate entity extraction accuracy."""
    total = len(dataset)
    found = 0
    not_found = []
    
    print("\n" + "="*70)
    print("ENTITY EXTRACTION EVALUATION")
    print("="*70)
    
    for item in dataset:
        text = item['text']
        ground_truth_entity = item['entity_text']
        
        # Extract entities with spaCy
        entities = extract_entities_spacy(text, nlp)
        entity_texts = [e['text'] for e in entities]
        
        # Check if ground truth entity was found
        if ground_truth_entity in entity_texts:
            found += 1
        else:
            # Check for partial match
            partial_match = any(ground_truth_entity in e or e in ground_truth_entity for e in entity_texts)
            if partial_match:
                found += 1
            else:
                not_found.append({
                    'text': text,
                    'expected': ground_truth_entity,
                    'found': entity_texts
                })
    
    precision = (found / total) * 100 if total > 0 else 0
    
    print(f"\n✅ Entities Found: {found}/{total} ({precision:.1f}%)")
    print(f"❌ Entities Missed: {len(not_found)}")
    
    if not_found and len(not_found) <= 5:
        print("\n🔍 Examples of Missed Entities:")
        for item in not_found[:3]:
            print(f"  Expected: '{item['expected']}'")
            print(f"  Text: {item['text'][:70]}...")
            print(f"  Found: {item['found']}\n")
    
    return {'total': total, 'found': found, 'precision': precision}


def evaluate_entity_classification(dataset, nlp):
    """Evaluate entity classification accuracy."""
    print("\n" + "="*70)
    print("ENTITY CLASSIFICATION EVALUATION")
    print("="*70)
    
    results_by_category = defaultdict(lambda: {'total': 0, 'correct': 0})
    results_by_known = {'known': {'total': 0, 'correct': 0}, 'unknown': {'total': 0, 'correct': 0}}
    
    for item in dataset:
        text = item['text']
        ground_truth_entity = item['entity_text']
        ground_truth_label = item['entity_label']
        ground_truth_category = item['entity_category']
        is_well_known = item['is_well_known']
        
        # Extract entities with spaCy
        entities = extract_entities_spacy(text, nlp)
        
        # Find the matching entity
        matched_entity = None
        for e in entities:
            if ground_truth_entity in e['text'] or e['text'] in ground_truth_entity:
                matched_entity = e
                break
        
        # Track stats
        results_by_category[ground_truth_category]['total'] += 1
        known_key = 'known' if is_well_known else 'unknown'
        results_by_known[known_key]['total'] += 1
        
        if matched_entity and matched_entity['label'] == ground_truth_label:
            results_by_category[ground_truth_category]['correct'] += 1
            results_by_known[known_key]['correct'] += 1
    
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
    
    print("\n" + "="*70)
    print(f"🎯 OVERALL CLASSIFICATION ACCURACY: {overall_accuracy:.1f}%")
    print("="*70)
    
    return {'overall_accuracy': overall_accuracy}


def benchmark_speed(dataset, nlp, sample_size=50):
    """Benchmark processing speed."""
    print("\n" + "="*70)
    print("SPEED BENCHMARK")
    print("="*70)
    
    sample = dataset[:min(sample_size, len(dataset))]
    texts = [item['text'] for item in sample]
    
    start_time = time.time()
    for text in texts:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / len(texts)
    
    print(f"\n⏱️  Processed {len(texts)} documents in {total_time:.2f} seconds")
    print(f"⚡  Average time per document: {avg_time*1000:.1f} ms")
    print(f"🚀  Throughput: {len(texts)/total_time:.1f} documents/second")
    
    return {'avg_time_ms': avg_time * 1000, 'throughput': len(texts) / total_time}


def main():
    print("="*70)
    print("PHASE 1: ENTITY CLASSIFICATION EVALUATION - SPACY BASELINE")
    print("="*70)
    print(f"Python: {sys.executable}")
    print(f"spaCy version: {spacy.__version__}\n")
    
    # Load dataset
    dataset_path = "banking_ner_dataset.csv"
    print(f"📂 Loading dataset: {dataset_path}")
    
    try:
        dataset = load_ground_truth_dataset(dataset_path)
        print(f"✅ Loaded {len(dataset)} labeled examples")
        
        unique_entities = len(set(item['entity_text'] for item in dataset))
        unique_categories = len(set(item['entity_category'] for item in dataset))
        well_known = sum(1 for item in dataset if item['is_well_known'])
        unknown = len(dataset) - well_known
        
        print(f"   - {unique_entities} unique entities")
        print(f"   - {unique_categories} entity categories")
        print(f"   - {well_known} well-known entities")
        print(f"   - {unknown} unknown/local entities")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {dataset_path}")
        sys.exit(1)
    
    # Load spaCy model
    print("\n🔧 Loading spaCy model: en_core_web_md")
    try:
        nlp = spacy.load('en_core_web_md')
        print("✅ Model loaded successfully")
    except OSError:
        print("❌ Error: Model not found. Install: python -m spacy download en_core_web_md")
        sys.exit(1)
    
    # Run evaluations
    extraction = evaluate_entity_extraction(dataset, nlp)
    classification = evaluate_entity_classification(dataset, nlp)
    speed = benchmark_speed(dataset, nlp)
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY - SPACY BASELINE RESULTS")
    print("="*70)
    print(f"Entity Extraction Accuracy:     {extraction['precision']:.1f}%")
    print(f"Entity Classification Accuracy: {classification['overall_accuracy']:.1f}%")
    print(f"Processing Speed:               {speed['avg_time_ms']:.1f} ms/document")
    print(f"Throughput:                     {speed['throughput']:.1f} docs/second")
    print("\n✅ Phase 1 baseline evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
