"""
Working Capital Optimization Demo with Gemini Explanations

This script demonstrates the full working capital optimization system using:
1. Sample data generation
2. AP optimization
3. AR optimization
4. Integrated working capital optimization
5. Gemini-powered explanations

Make sure to set up your environment variables before running:
- GOOGLE_API_KEY for Gemini
"""
import os
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import argparse


def load_real_data(filepath, ar_mode=False):
    """
    Load and flatten real AP/AR JSON data for optimization.
    Each invoice is enriched with its parent business entity fields.
    If ar_mode is True, map 'balance_original' to 'amount' for AR optimizer compatibility.
    Returns a flat list of records (or a DataFrame if needed).
    """
    print(f"[DEBUG] Loading real data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            print(f"[DEBUG] First 200 chars of file: {content[:200]}")
            entities = json.loads(content)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON from {filepath}: {e}")
        raise
    flat_records = []
    for entity in entities:
        entity_fields = {k: v for k, v in entity.items() if k != 'due_invoices'}
        for invoice in entity.get('due_invoices', []):
            record = dict(entity_fields)  # Copy entity info
            record.update(invoice)        # Add invoice info
            if ar_mode:
                record['amount'] = invoice.get('balance_original', 0)
            flat_records.append(record)
    return flat_records

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from optimization.ap_optimizer import APOptimizer
from optimization.ar_optimizer import AROptimizer
from ai.gemini_explanation import GeminiExplanationEngine

def setup_args():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(description='Run AP/AR optimization demo')
    parser.add_argument('--ap', action='store_true', help='Run AP optimization')
    parser.add_argument('--ar', action='store_true', help='Run AR optimization')
    parser.add_argument('--all', action='store_true', help='Run all optimizations')
    parser.add_argument('--explain', action='store_true', help='Generate explanations')
    parser.add_argument('--num-ap', type=int, default=30, help='Number of AP invoices')
    parser.add_argument('--num-ar', type=int, default=30, help='Number of AR invoices')
    parser.add_argument('--horizon', type=int, default=90, help='Time horizon in days')
    parser.add_argument('--cash', type=float, default=250000, help='Initial cash balance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--ap_data', type=str, default=None, help='Path to real AP data JSON')
    parser.add_argument('--ar_data', type=str, default=None, help='Path to real AR data JSON')
    return parser.parse_args()

def run_ap_optimization(data, args):
    """Run the AP optimization model"""
    print("\n=== Running Accounts Payable Optimization ===")

    # If data is a path to a real JSON, flatten it
    if isinstance(data, str) and data.endswith('.json'):
        data = {"ap_invoices": load_real_data(data)}

    # Configure the AP optimizer
    config = {
        "optimization_mode": "balanced",  # cost, supplier, cash, or custom
        "horizon": args.horizon,
        "max_borrowing": args.cash * 1.5,  # Allow borrowing up to 150% of initial cash
        "borrowing_rate": 0.0001  # Daily rate (approx 3.65% annually)
    }
    
    # Initialize and run the optimizer
    optimizer = APOptimizer(config)
    processed_data = optimizer.preprocess_data(data)
    optimizer.build_model(processed_data)
    results = optimizer.solve()
    
    # Print summary of results
    if results["status"] == "optimal" or results["status"] == "feasible_non_optimal":
        print(f"Optimization successful! Objective value: {results['objective_value']:.2f}")
        print("\nKey Metrics:")
        metrics = results["key_metrics"]
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
            
        print("\nPayment Schedule Summary:")
        payment_schedule = results["payment_schedule"]
        
        # Group by payment timing
        early_with_discount = sum(1 for p in payment_schedule if p.get("discount_captured"))
        early_no_discount = sum(1 for p in payment_schedule if p.get("days_early", 0) > 0 and not p.get("discount_captured"))
        on_time = sum(1 for p in payment_schedule if p.get("days_early", 0) == 0 and p.get("days_late", 0) > 0)
        late = sum(1 for p in payment_schedule if p.get("days_late", 0) > 0)
        
        total = len(payment_schedule)
        print(f"  Total invoices: {total}")
        print(f"  Early with discount: {early_with_discount} ({early_with_discount/total*100:.1f}%)")
        print(f"  Early without discount: {early_no_discount} ({early_no_discount/total*100:.1f}%)")
        print(f"  On time: {on_time} ({on_time/total*100:.1f}%)")
        print(f"  Late: {late} ({late/total*100:.1f}%)")
        
        # Save results
        results_path = os.path.join(args.output, "ap_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Add priority calculation for each payment schedule item
        def calculate_priority(item):
            """Calculate priority score based on days overdue and balance"""
            # Normalize days overdue to a scale of 0-1 (assuming max overdue days is 30)
            normalized_days = min(item['days_overdue'] / 30, 1.0)
            # Normalize balance to a scale of 0-1 (using log to reduce impact of very large balances)
            normalized_balance = min(1.0, item['total_balance_original'] / 10000000)  # Normalize to millions
            
            # Combine both factors with weights (adjust weights as needed)
            priority_score = (0.7 * normalized_days + 0.3 * normalized_balance)
            return priority_score

        # Add priority to each payment schedule item and sort before saving
        payment_schedule = results["payment_schedule"]
        for item in payment_schedule:
            item['priority'] = calculate_priority(item)
        payment_schedule = sorted(payment_schedule, key=lambda x: x['priority'], reverse=True)
        results["payment_schedule"] = payment_schedule
        
        # Print top 5 priority items
        print("\nTop 5 Priority Items:")
        for item in payment_schedule[:5]:
            print(f"  {item['business_entity']}: Priority {item['priority']:.2f} (Days overdue: {item['days_overdue']}, Balance: {item['total_balance_original']:.2f})")
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        if results["status"] == "optimal" or results["status"] == "feasible_non_optimal":
            print(f"Optimization successful! Objective value: {results['objective_value']:.2f}")
            print("\nKey Metrics:")
            metrics = results["key_metrics"]
            for key, value in metrics.items():
                print(f"  {key}: {value:.2f}")
        else:
            print(f"Optimization failed: {results.get('error', 'Unknown error')}")
            return results
        
        # Save results before returning
        results_path = os.path.join(args.output, "ap_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        return results
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def run_ar_optimization(data, args):
    """Run the AR optimization model"""
    print("\n=== Running Accounts Receivable Optimization ===")

    # If data is a path to a real JSON, flatten it
    if isinstance(data, str) and data.endswith('.json'):
        data = {"ar_invoices": load_real_data(data, ar_mode=True)}

    # Configure the AR optimizer with more lenient settings
    config = {
        "optimization_mode": "cash_flow",  # Start with cash flow optimization
        "horizon": args.horizon,
        "max_financing": args.cash * 3.0,  # Allow financing up to 300% of initial cash
        "financing_rate": 0.00005,  # Lower daily rate (approx 1.83% annually)
        "collection_actions": [
            "reminder",  # Low cost, low impact
            "call",      # Medium cost, medium impact
            "escalate"   # High cost, high impact
        ],
        "collection_weight": 1.0,
        "financing_weight": 0.3,  # Lower weight on financing to reduce its impact
        "relationship_weight": 0.1,  # Lower weight on relationship to make it more flexible
        "transaction_weight": 0.1,  # Lower weight on transaction costs
        "min_collection_probability": 0.0,  # Allow lower collection probabilities
        "max_collection_probability": 1.0,  # Allow higher collection probabilities
        "allow_partial_payments": True  # Allow partial payments on invoices
    }
    
    print(f"AR Optimization Configuration:")
    print(f"- Mode: {config['optimization_mode']}")
    print(f"- Horizon: {config['horizon']} days")
    print(f"- Max financing: {config['max_financing']:.2f}")
    print(f"- Financing rate: {config['financing_rate']*100:.4f}% daily")
    
    # Initialize and run the optimizer
    optimizer = AROptimizer(config)
    processed_data = optimizer.preprocess_data(data)
    optimizer.build_model(processed_data)
    results = optimizer.solve()
    
    # Print summary of results
    if results.get("status") in ["optimal", "feasible_non_optimal"]:
        print(f"Optimization successful! Objective value: {results.get('objective_value', 0):.2f}")
        print("\nKey Metrics:")
        metrics = results.get("key_metrics", {})
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
        
        print("\nCollection Strategy Summary:")
        action_plan = results.get("action_plan", [])
        
        # Group actions by type
        action_counts = {}
        for action in action_plan:
            action_type = action.get("action")
            if action_type not in action_counts:
                action_counts[action_type] = 0
            action_counts[action_type] += 1
        
        print(f"  Total actions recommended: {len(action_plan)}")
        for action_type, count in action_counts.items():
            print(f"  {action_type}: {count}")
        
        # Show collections by timing
        ar_decisions = results.get("ar_decisions", [])
        timing = results.get("collection_timing", {})
        before_due = timing.get("before_due", 0)
        after_due = timing.get("after_due", 0)
        total = before_due + after_due if (before_due + after_due) > 0 else 1
        print("\nExpected Collection Timing:")
        print(f"  Before due date: {before_due} ({before_due/total*100:.1f}%)")
        print(f"  After due date: {after_due} ({after_due/total*100:.1f}%)")
        
        # Save results
        results_path = os.path.join(args.output, "ar_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def generate_explanations(results, args, optimization_type="ap"):
    """Generate explanations for optimization results using Gemini"""
    print("\n=== Generating Gemini-powered Explanations ===")
    
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable is not set. Cannot generate explanations.")
        return
    
    try:
        # Initialize the explanation engine
        engine = GeminiExplanationEngine()
        
        # Generate general explanation
        print("\n1. General Explanation:")
        general_explanation = engine.generate_explanation(
            results,
            explanation_type="general"
        )
        print(general_explanation)
        
        # Generate decision explanation for a specific invoice
        if optimization_type == "ap" and "ap_decisions" in results:
            # Find an interesting invoice (with discount or late payment)
            ap_decisions = results["ap_decisions"]
            interesting_decision = next(
                (d for d in ap_decisions if d.get("payment_timing") == "early_with_discount"),
                next(
                    (d for d in ap_decisions if d.get("payment_timing") == "late"),
                    ap_decisions[0] if ap_decisions else None
                )
            )
            
            if interesting_decision:
                invoice_id = interesting_decision["invoice_id"]
                print(f"\n2. Decision Explanation for invoice {invoice_id}:")
                decision_explanation = engine.generate_explanation(
                    results,
                    explanation_type="decision",
                    specific_entity=invoice_id
                )
                print(decision_explanation)
                
        elif optimization_type == "ar" and "ar_decisions" in results:
            # Find an interesting invoice (with multiple actions)
            ar_decisions = results["ar_decisions"]
            interesting_decision = next(
                (d for d in ar_decisions if len(d.get("actions", [])) > 1),
                ar_decisions[0] if ar_decisions else None
            )
            
            if interesting_decision:
                invoice_id = interesting_decision["invoice_id"]
                print(f"\n2. Decision Explanation for invoice {invoice_id}:")
                decision_explanation = engine.generate_explanation(
                    results,
                    explanation_type="decision",
                    specific_entity=invoice_id
                )
                print(decision_explanation)
                
        # Generate root cause analysis
        print("\n3. Root Cause Analysis for Working Capital Position:")
        questions = {
            "ap": "What factors are driving our accounts payable strategy?",
            "ar": "What factors are influencing our accounts receivable collection patterns?"
        }
        root_cause_explanation = engine.generate_explanation(
            results,
            explanation_type="root_cause",
            specific_question=questions[optimization_type]
        )
        print(root_cause_explanation)
        
        # Save explanations
        explanations = {
            "general_explanation": general_explanation,
            "decision_explanation": decision_explanation if 'decision_explanation' in locals() else "Not generated",
            "root_cause_explanation": root_cause_explanation,
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "optimization_type": optimization_type
            }
        }
        
        explanations_path = os.path.join(args.output, f"{optimization_type}_explanations.json")
        with open(explanations_path, "w") as f:
            json.dump(explanations, f, indent=2)
        
        print(f"Explanations saved to: {explanations_path}")
        
    except Exception as e:
        print(f"Error generating explanations: {str(e)}")

def main():
    """Main demo script"""
    # Parse command line arguments
    args = setup_args()
    
    print("=== AP/AR Optimization Demo with Gemini Explanations ===")
    
    # Default to running all if no specific optimization is selected
    if not (args.ap or args.ar or args.all):
        args.all = True
    
    # Data selection logic
    ap_data = r"D:\Download\code_math_model\data\data_AP.json"
    ar_data = r"D:\Download\code_math_model\data\data_AR.json"
    if args.ap_data:
        ap_data = args.ap_data
    if args.ar_data:
        ar_data = args.ar_data

    # Load actual AP/AR data from JSON files if they are strings (file paths)
    if isinstance(ap_data, str):
        ap_data = load_real_data(ap_data)
    if isinstance(ar_data, str):
        ar_data = load_real_data(ar_data, ar_mode=True)

    # Run optimizations
    ap_results = None
    ar_results = None

    if args.ap or args.all:
        if ap_data:
            ap_results = run_ap_optimization({"ap_invoices": ap_data}, args)
        if args.explain and ap_results and ap_results.get("status") in ["optimal", "feasible_non_optimal"]:
            generate_explanations(ap_results, args, "ap")

    if args.ar or args.all:
        if ar_data:
            ar_results = run_ar_optimization({"ar_invoices": ar_data}, args)
        if args.explain and ar_results and ar_results.get("status") in ["optimal", "feasible_non_optimal"]:
            generate_explanations(ar_results, args, "ar")
    
    print("\nDemo complete! The optimization system successfully:")
    if ap_results and ap_results.get("status") in ["optimal", "feasible_non_optimal"]:
        print("✓ Optimized accounts payable schedule")
    if ar_results and ar_results.get("status") in ["optimal", "feasible_non_optimal"]:
        print("✓ Optimized accounts receivable collection strategy")
    if args.explain and (
        (ap_results and ap_results.get("status") in ["optimal", "feasible_non_optimal"]) or
        (ar_results and ar_results.get("status") in ["optimal", "feasible_non_optimal"])
    ):
        print("✓ Generated Gemini-powered explanations for the results")
    
    print(f"\nAll results and explanations saved to: {args.output}/")

if __name__ == "__main__":
    main()
