"""
Compare All NLP Approaches
Runs all evaluation scripts and creates a comprehensive comparison report.
"""

import sys
import json
import subprocess
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("❌ Missing 'pandas'. Install with: pip install pandas")
    sys.exit(1)


def run_evaluation(script_name, description):
    """Run an evaluation script and return results."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {description} timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def load_results(filename):
    """Load results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Results file not found: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"⚠️  Invalid JSON in: {filename}")
        return None


def create_comparison_table(results_dict):
    """Create comparison table from all results."""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    
    # Prepare data
    data = []
    for approach, results in results_dict.items():
        if results:
            data.append({
                'Approach': approach,
                'Model': results.get('model', 'N/A'),
                'Overall Accuracy': f"{results.get('overall_accuracy', 0):.1f}%",
                'Known Entities': f"{results.get('by_known', {}).get('known', {}).get('correct', 0)}/{results.get('by_known', {}).get('known', {}).get('total', 0)}",
                'Unknown Entities': f"{results.get('by_known', {}).get('unknown', {}).get('correct', 0)}/{results.get('by_known', {}).get('unknown', {}).get('total', 0)}",
                'Avg Time (ms)': f"{results.get('avg_time_ms', 0):.1f}",
                'Throughput': f"{results.get('throughput', 0):.1f}/sec"
            })
    
    if not data:
        print("❌ No results available for comparison")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print table
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    csv_filename = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Comparison table saved to: {csv_filename}")
    
    return df


def print_recommendations(results_dict):
    """Print recommendations based on results."""
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    spacy_results = results_dict.get('spaCy')
    zero_shot_results = results_dict.get('Zero-Shot')
    
    if spacy_results and zero_shot_results:
        spacy_acc = spacy_results.get('overall_accuracy', 0)
        zero_shot_acc = zero_shot_results.get('overall_accuracy', 0)
        spacy_speed = spacy_results.get('avg_time_ms', 0)
        zero_shot_speed = zero_shot_results.get('avg_time_ms', 0)
        
        print("\n📈 Accuracy Comparison:")
        print(f"   spaCy: {spacy_acc:.1f}%")
        print(f"   Zero-Shot: {zero_shot_acc:.1f}%")
        if zero_shot_acc > spacy_acc:
            print(f"   ✅ Zero-Shot is {zero_shot_acc - spacy_acc:.1f}% more accurate!")
        
        print("\n⚡ Speed Comparison:")
        print(f"   spaCy: {spacy_speed:.1f} ms")
        print(f"   Zero-Shot: {zero_shot_speed:.1f} ms")
        if spacy_speed < zero_shot_speed:
            speedup = zero_shot_speed / spacy_speed
            print(f"   ✅ spaCy is {speedup:.1f}x faster!")
        
        print("\n🎯 Recommended Approach:")
        if spacy_acc >= 85 and spacy_speed < 10:
            print("   ✅ Use spaCy for production (fast + accurate enough)")
        elif zero_shot_acc >= 85:
            print("   ✅ Use Zero-Shot for high accuracy needs")
        else:
            print("   ⭐ Use HYBRID approach:")
            print("      1. spaCy for fast extraction")
            print("      2. Zero-Shot for unknown entities")
            print("      3. Knowledge base for frequent entities")
            print(f"      Expected: 88-92% accuracy, 50-100ms speed")
    
    print("\n" + "="*70)


def main():
    print("="*70)
    print("COMPREHENSIVE NLP APPROACH COMPARISON")
    print("="*70)
    print("This script will run all evaluation approaches and compare results")
    print("="*70)
    
    # Define evaluations to run
    evaluations = [
        ('evaluate_spacy.py', 'spaCy Baseline', 'results_spacy.json', 'spaCy'),
        ('evaluate_zero_shot.py', 'Zero-Shot Classification', 'results_zero_shot.json', 'Zero-Shot'),
        ('evaluate_finbert.py', 'FinBERT (Finance-Specific)', 'results_finbert.json', 'FinBERT')
    ]
    
    results_dict = {}
    
    # Run each evaluation
    for script, description, result_file, key in evaluations:
        success = run_evaluation(script, description)
        if success:
            results = load_results(result_file)
            if results:
                results_dict[key] = results
    
    # Create comparison table
    if results_dict:
        df = create_comparison_table(results_dict)
        print_recommendations(results_dict)
        
        print("\n" + "="*70)
        print("✅ COMPARISON COMPLETE!")
        print("="*70)
        print("\n📂 Generated Files:")
        print("   - results_spacy.json (spaCy baseline results)")
        print("   - results_zero_shot.json (Zero-Shot results)")
        print("   - results_finbert.json (FinBERT results)")
        print("   - comparison_results_*.csv (comparison table)")
        print("="*70)
    else:
        print("\n❌ No results available for comparison")
        print("   Make sure the evaluation scripts run successfully")


if __name__ == "__main__":
    main()
