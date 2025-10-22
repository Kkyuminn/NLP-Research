"""
Shared Utilities for NLP Evaluation
Common functions used across different NLP evaluation scripts.
"""

import csv
from collections import defaultdict


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


def print_dataset_stats(dataset):
    """Print statistics about the dataset."""
    unique_entities = len(set(item['entity_text'] for item in dataset))
    unique_categories = len(set(item['entity_category'] for item in dataset))
    well_known = sum(1 for item in dataset if item['is_well_known'])
    unknown = len(dataset) - well_known
    
    print(f"✅ Loaded {len(dataset)} labeled examples")
    print(f"   - {unique_entities} unique entities")
    print(f"   - {unique_categories} entity categories")
    print(f"   - {well_known} well-known entities")
    print(f"   - {unknown} unknown/local entities")


def print_results_by_category(results_by_category, title="Accuracy by Entity Category"):
    """Print results grouped by entity category."""
    print(f"\n📊 {title}:")
    print(f"{'Category':<35} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 70)
    
    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{category:<35} {stats['total']:<8} {stats['correct']:<8} {accuracy:.1f}%")


def print_results_by_familiarity(results_by_known, title="Accuracy by Entity Familiarity"):
    """Print results grouped by known/unknown entities."""
    print(f"\n📊 {title}:")
    print(f"{'Type':<35} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 70)
    
    for known_type in ['known', 'unknown']:
        stats = results_by_known[known_type]
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        label = "Well-known entities" if known_type == 'known' else "Unknown/Local entities"
        print(f"{label:<35} {stats['total']:<8} {stats['correct']:<8} {accuracy:.1f}%")


def calculate_overall_accuracy(results_by_category):
    """Calculate overall accuracy from results."""
    total_correct = sum(cat['correct'] for cat in results_by_category.values())
    total_items = sum(cat['total'] for cat in results_by_category.values())
    return (total_correct / total_items * 100) if total_items > 0 else 0


def save_results_to_json(results, filename):
    """Save evaluation results to JSON file."""
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {filename}")
