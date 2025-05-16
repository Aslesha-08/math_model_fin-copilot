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

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data.sample_data_generator import SampleDataGenerator
from optimization.ap_optimizer import APOptimizer
from optimization.ar_simple import SimpleAROptimizer
from optimization.wc_simple import SimpleWorkingCapitalOptimizer
from ai.gemini_explanation import GeminiExplanationEngine

def setup_args():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(description='Run working capital optimization demo')
    parser.add_argument('--ap', action='store_true', help='Run AP optimization')
    parser.add_argument('--ar', action='store_true', help='Run AR optimization')
    parser.add_argument('--wc', action='store_true', help='Run WC optimization')
    parser.add_argument('--all', action='store_true', help='Run all optimizations')
    parser.add_argument('--explain', action='store_true', help='Generate explanations')
    parser.add_argument('--num-ap', type=int, default=30, help='Number of AP invoices')
    parser.add_argument('--num-ar', type=int, default=30, help='Number of AR invoices')
    parser.add_argument('--horizon', type=int, default=90, help='Time horizon in days')
    parser.add_argument('--cash', type=float, default=250000, help='Initial cash balance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    return parser.parse_args()

def generate_sample_data(args):
    """Generate sample data using the SampleDataGenerator"""
    print("Generating sample data...")
    generator = SampleDataGenerator(
        num_ap_invoices=args.num_ap, 
        num_ar_invoices=args.num_ar, 
        horizon=args.horizon, 
        initial_cash=args.cash,
        seed=args.seed
    )
    
    data = generator.generate_data()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save the data
    data_path = os.path.join(args.output, "sample_data.json")
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated sample data with {len(data['ap_invoices'])} AP invoices and {len(data['ar_invoices'])} AR invoices")
    print(f"Data saved to: {data_path}")
    
    return data

def run_ap_optimization(data, args):
    """Run the AP optimization model"""
    print("\n=== Running Accounts Payable Optimization ===")
    
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
        on_time = sum(1 for p in payment_schedule if p.get("days_early", 0) == 0 and p.get("days_late", 0) == 0)
        late = sum(1 for p in payment_schedule if p.get("days_late", 0) > 0)
        
        total = len(payment_schedule)
        print(f"  Total invoices: {total}")
        print(f"  Early with discount: {early_with_discount} ({early_with_discount/total*100:.1f}%)")
        print(f"  Early without discount: {early_no_discount} ({early_no_discount/total*100:.1f}%)")
        print(f"  On time: {on_time} ({on_time/total*100:.1f}%)")
        print(f"  Late: {late} ({late/total*100:.1f}%)")
        
        # Save results
        results_path = os.path.join(args.output, "ap_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def run_ar_optimization(data, args):
    """Run the AR optimization model"""
    print("\n=== Running Accounts Receivable Optimization ===")
    
    # Configure the AR optimizer
    config = {
        "optimization_mode": "balanced",  # cash_flow, customer, financing, or custom
        "horizon": args.horizon,
        "max_financing": args.cash * 1.5,  # Allow financing up to 150% of initial cash
        "financing_rate": 0.0001,  # Daily rate (approx 3.65% annually)
        "collection_actions": [
            "reminder",
            "call",
            "escalate",
            "discount_offer",
            "late_fee"
        ]
    }
    
    # Initialize and run the optimizer
    optimizer = SimpleAROptimizer(config)
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
        before_due = sum(1 for d in ar_decisions if d.get("collection_timing") == "before_due")
        after_due = sum(1 for d in ar_decisions if d.get("collection_timing") == "after_due")
        
        total = len(ar_decisions)
        if total > 0:
            print(f"\nExpected Collection Timing:")
            print(f"  Before due date: {before_due} ({before_due/total*100:.1f}%)")
            print(f"  After due date: {after_due} ({after_due/total*100:.1f}%)")
        
        # Save results
        results_path = os.path.join(args.output, "ar_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def run_wc_optimization(data, args):
    """Run the integrated Working Capital optimization model"""
    print("\n=== Running Integrated Working Capital Optimization ===")
    
    # Configure the WC optimizer
    config = {
        "optimization_mode": "balanced",  # cash_flow, balanced, relationship, cost, or custom
        "horizon": args.horizon,
        "max_financing": args.cash * 1.5,  # Allow financing up to 150% of initial cash
        "financing_rate": 0.0001,  # Daily rate (approx 3.65% annually)
        # AP-specific weights
        "ap_weights": {
            "discount_weight": 0.9,
            "penalty_weight": 1.0,
            "relationship_weight": 0.7,
            "cash_weight": 0.8
        },
        # AR-specific weights
        "ar_weights": {
            "collection_weight": 0.9,
            "financing_weight": 0.8,
            "relationship_weight": 0.7,
            "transaction_weight": 0.6
        }
    }
    
    # Initialize and run the optimizer
    optimizer = SimpleWorkingCapitalOptimizer(config)
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
            
        print("\nWorking Capital Management Summary:")
        if "net_working_capital" in metrics:
            print(f"  Net Working Capital: ${metrics['net_working_capital']:.2f}")
        if "cash_conversion_cycle" in metrics:
            print(f"  Cash Conversion Cycle: {metrics['cash_conversion_cycle']:.1f} days")
        if "max_financing_needed" in metrics:
            print(f"  Maximum External Financing: ${metrics['max_financing_needed']:.2f}")
            
        # AP summary
        print("\nAccounts Payable Summary:")
        ap_decisions = results.get("ap_decisions", [])
        early_with_discount = sum(1 for p in ap_decisions if p.get("payment_timing") == "early_with_discount")
        early_no_discount = sum(1 for p in ap_decisions if p.get("payment_timing") == "early_no_discount")
        on_time = sum(1 for p in ap_decisions if p.get("payment_timing") == "on_time")
        late = sum(1 for p in ap_decisions if p.get("payment_timing") == "late")
        
        total_ap = len(ap_decisions)
        if total_ap > 0:
            print(f"  Total AP invoices: {total_ap}")
            print(f"  Early with discount: {early_with_discount} ({early_with_discount/total_ap*100:.1f}%)")
            print(f"  Early without discount: {early_no_discount} ({early_no_discount/total_ap*100:.1f}%)")
            print(f"  On time: {on_time} ({on_time/total_ap*100:.1f}%)")
            print(f"  Late: {late} ({late/total_ap*100:.1f}%)")
        
        # AR summary
        print("\nAccounts Receivable Summary:")
        ar_decisions = results.get("ar_decisions", [])
        
        # Group actions by type
        action_counts = {}
        for decision in ar_decisions:
            for action in decision.get("actions", []):
                if action not in action_counts:
                    action_counts[action] = 0
                action_counts[action] += 1
        
        for action_type, count in action_counts.items():
            print(f"  {action_type}: {count}")
        
        # Show collections by timing
        before_due = sum(1 for d in ar_decisions if d.get("collection_timing") == "before_due")
        after_due = sum(1 for d in ar_decisions if d.get("collection_timing") == "after_due")
        
        total_ar = len(ar_decisions)
        if total_ar > 0:
            print(f"\nExpected Collection Timing:")
            print(f"  Before due date: {before_due} ({before_due/total_ar*100:.1f}%)")
            print(f"  After due date: {after_due} ({after_due/total_ar*100:.1f}%)")
        
        # Save results
        results_path = os.path.join(args.output, "wc_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def generate_explanations(results, args, optimization_type="wc"):
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
                
        elif optimization_type == "wc":
            # For WC, pick either an AP or AR decision
            if "ap_decisions" in results and results["ap_decisions"]:
                invoice_id = results["ap_decisions"][0]["invoice_id"]
            elif "ar_decisions" in results and results["ar_decisions"]:
                invoice_id = results["ar_decisions"][0]["invoice_id"]
            else:
                invoice_id = None
                
            if invoice_id:
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
            "ar": "What factors are influencing our accounts receivable collection patterns?",
            "wc": "What factors are driving our overall working capital position?"
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
    
    print("=== Working Capital Optimization Demo with Gemini Explanations ===")
    
    # Default to running all if no specific optimization is selected
    if not (args.ap or args.ar or args.wc or args.all):
        args.all = True
    
    # Generate sample data
    data = generate_sample_data(args)
    
    # Run optimizations
    ap_results = None
    ar_results = None
    wc_results = None
    
    if args.ap or args.all:
        ap_results = run_ap_optimization(data, args)
        if args.explain and ap_results and ap_results.get("status") in ["optimal", "feasible_non_optimal"]:
            generate_explanations(ap_results, args, "ap")
    
    if args.ar or args.all:
        ar_results = run_ar_optimization(data, args)
        if args.explain and ar_results and ar_results.get("status") in ["optimal", "feasible_non_optimal"]:
            generate_explanations(ar_results, args, "ar")
    
    if args.wc or args.all:
        # Make sure we have both AP and AR data
        if "ap_invoices" not in data or "ar_invoices" not in data:
            print("Error: Both AP and AR data are required for working capital optimization.")
        else:
            wc_results = run_wc_optimization(data, args)
            if args.explain and wc_results and wc_results.get("status") in ["optimal", "feasible_non_optimal"]:
                generate_explanations(wc_results, args, "wc")
    
    print("\nDemo complete! The working capital optimization system successfully:")
    if ap_results and ap_results.get("status") in ["optimal", "feasible_non_optimal"]:
        print("✓ Optimized accounts payable schedule")
    if ar_results and ar_results.get("status") in ["optimal", "feasible_non_optimal"]:
        print("✓ Optimized accounts receivable collection strategy")
    if wc_results and wc_results.get("status") in ["optimal", "feasible_non_optimal"]:
        print("✓ Integrated both for holistic working capital management")
    if args.explain and (
        (ap_results and ap_results.get("status") in ["optimal", "feasible_non_optimal"]) or
        (ar_results and ar_results.get("status") in ["optimal", "feasible_non_optimal"]) or
        (wc_results and wc_results.get("status") in ["optimal", "feasible_non_optimal"])
    ):
        print("✓ Generated Gemini-powered explanations for the results")
    
    print(f"\nAll results and explanations saved to: {args.output}/")

if __name__ == "__main__":
    main()
