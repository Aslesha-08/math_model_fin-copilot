"""
Working Capital Optimization Demo Script

This script demonstrates the end-to-end functionality of the working capital optimization system,
including the AP, AR, and integrated WC optimizers, as well as the explanation engine.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local imports
from optimization.ap_optimizer import APOptimizer
from optimization.ar_optimizer import AROptimizer
from optimization.wc_optimizer import WorkingCapitalOptimizer
from ai.explanation import ExplanationEngine

def generate_sample_ap_data(num_invoices=20):
    """Generate sample AP invoice data"""
    invoices = []
    
    # Current day is day 1
    current_day = 1
    
    # Supplier categories with different priorities
    suppliers = {
        "strategic": ["Supplier_A", "Supplier_B", "Supplier_C"],
        "regular": ["Supplier_D", "Supplier_E", "Supplier_F", "Supplier_G"],
        "non_critical": ["Supplier_H", "Supplier_I", "Supplier_J", "Supplier_K"]
    }
    
    # Priority mapping
    priority_map = {
        "strategic": "high",
        "regular": "medium",
        "non_critical": "low"
    }
    
    for i in range(num_invoices):
        # Determine supplier category
        if i < num_invoices * 0.3:
            category = "strategic"
        elif i < num_invoices * 0.7:
            category = "regular"
        else:
            category = "non_critical"
            
        # Select random supplier from category
        supplier_id = np.random.choice(suppliers[category])
        
        # Generate invoice amount (larger for strategic suppliers)
        if category == "strategic":
            amount = np.random.uniform(10000, 50000)
        elif category == "regular":
            amount = np.random.uniform(5000, 15000)
        else:
            amount = np.random.uniform(1000, 7000)
        
        # Generate due date (15-45 days out)
        due_date = current_day + np.random.randint(15, 46)
        
        # Early payment discount (more common for non-strategic suppliers)
        has_discount = np.random.random() < (0.2 if category == "strategic" else 0.5)
        if has_discount:
            discount_date = due_date - np.random.randint(7, 15)
            discount_rate = np.random.uniform(0.01, 0.03)
        else:
            discount_date = None
            discount_rate = 0.0
        
        # Late payment penalty (stricter for strategic suppliers)
        if category == "strategic":
            penalty_rate = np.random.uniform(0.001, 0.002)  # 0.1-0.2% per day
        else:
            penalty_rate = np.random.uniform(0.0005, 0.001)  # 0.05-0.1% per day
        
        # Create invoice
        invoice = {
            "id": f"AP{i+1:03d}",
            "amount": round(amount, 2),
            "due_date": due_date,
            "discount_date": discount_date,
            "discount_rate": discount_rate,
            "penalty_rate": penalty_rate,
            "supplier_id": supplier_id,
            "supplier_priority": priority_map[category]
        }
        
        invoices.append(invoice)
    
    # Generate cash position and forecasts
    cash_position = {
        "initial_balance": 100000,
        "min_balance": 50000
    }
    
    forecasts = {}
    
    # Day 1 has 0 inflows/outflows since starting balance is defined
    forecasts[1] = {
        "inflow": 0,
        "outflow": 0
    }
    
    # Generate some forecasted inflows and outflows for the next 30 days
    for day in range(2, 91):
        # More inflows around day 15 and 30 (payroll periods)
        if day % 15 == 0:
            inflow = np.random.uniform(80000, 120000)
        else:
            inflow = np.random.uniform(5000, 15000)
        
        # More outflows around day 10 and 25
        if day % 15 == 10:
            outflow = np.random.uniform(50000, 70000)
        else:
            outflow = np.random.uniform(3000, 8000)
            
        forecasts[day] = {
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2)
        }
    
    return {
        "ap_invoices": invoices,
        "cash_position": cash_position,
        "forecasts": forecasts
    }

def generate_sample_ar_data(num_invoices=20):
    """Generate sample AR invoice data"""
    invoices = []
    
    # Current day is day 1
    current_day = 1
    
    # Customer categories with different segments
    customers = {
        "strategic": ["Customer_A", "Customer_B", "Customer_C"],
        "regular": ["Customer_D", "Customer_E", "Customer_F", "Customer_G"],
        "small": ["Customer_H", "Customer_I", "Customer_J", "Customer_K"]
    }
    
    # Segment mapping
    segment_map = {
        "strategic": "high",
        "regular": "medium",
        "small": "low"
    }
    
    for i in range(num_invoices):
        # Determine customer category
        if i < num_invoices * 0.3:
            category = "strategic"
        elif i < num_invoices * 0.7:
            category = "regular"
        else:
            category = "small"
            
        # Select random customer from category
        customer_id = np.random.choice(customers[category])
        
        # Generate invoice amount (larger for strategic customers)
        if category == "strategic":
            amount = np.random.uniform(15000, 70000)
        elif category == "regular":
            amount = np.random.uniform(7000, 20000)
        else:
            amount = np.random.uniform(2000, 10000)
        
        # Generate issue date (1-15 days ago)
        issue_date = max(1, current_day - np.random.randint(1, 16))
        
        # Generate due date (30-60 days from issue date)
        due_date = issue_date + np.random.randint(30, 61)
        
        # Payment probability based on customer segment
        if category == "strategic":
            payment_probability = np.random.uniform(0.85, 0.95)
        elif category == "regular":
            payment_probability = np.random.uniform(0.75, 0.85)
        else:
            payment_probability = np.random.uniform(0.65, 0.75)
        
        # Create invoice
        invoice = {
            "id": f"AR{i+1:03d}",
            "amount": round(amount, 2),
            "issue_date": issue_date,
            "due_date": due_date,
            "payment_probability": payment_probability,
            "customer_id": customer_id,
            "customer_segment": segment_map[category]
        }
        
        invoices.append(invoice)
    
    # Generate cash position and forecasts (same as AP for simplicity)
    cash_position = {
        "initial_balance": 100000,
        "min_balance": 50000
    }
    
    forecasts = {}
    
    # Day 1 has 0 inflows/outflows since starting balance is defined
    forecasts[1] = {
        "inflow": 0,
        "outflow": 0
    }
    
    # Generate some forecasted inflows and outflows for the next 30 days
    for day in range(2, 91):
        # More inflows around day 15 and 30 (payroll periods)
        if day % 15 == 0:
            inflow = np.random.uniform(80000, 120000)
        else:
            inflow = np.random.uniform(5000, 15000)
        
        # More outflows around day 10 and 25
        if day % 15 == 10:
            outflow = np.random.uniform(50000, 70000)
        else:
            outflow = np.random.uniform(3000, 8000)
            
        forecasts[day] = {
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2)
        }
    
    return {
        "ar_invoices": invoices,
        "cash_position": cash_position,
        "forecasts": forecasts
    }

def run_ap_optimization(data):
    """Run the AP optimization model"""
    print("\n=== Running Accounts Payable Optimization ===")
    
    # Configure the AP optimizer
    config = {
        "optimization_mode": "cost",  # cost, supplier, cash, or custom
        "horizon": 90,
        "max_borrowing": 500000,
        "borrowing_rate": 0.0001  # Daily rate (approx 3.65% annually)
    }
    
    # Initialize and run the optimizer
    optimizer = APOptimizer(config)
    processed_data = optimizer.preprocess_data(data)
    optimizer.build_model(processed_data)
    results = optimizer.solve()
    
    # Print summary of results
    if results["status"] == "optimal":
        print(f"Optimization successful! Objective value: {results['objective_value']:.2f}")
        print("\nKey Metrics:")
        metrics = results["key_metrics"]
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
            
        print("\nPayment Schedule Summary:")
        payment_schedule = results["payment_schedule"]
        
        early_with_discount = sum(1 for p in payment_schedule if p["discount_captured"])
        early_no_discount = sum(1 for p in payment_schedule if p["days_early"] > 0 and not p["discount_captured"])
        on_time = sum(1 for p in payment_schedule if p["days_early"] == 0 and p["days_late"] == 0)
        late = sum(1 for p in payment_schedule if p["days_late"] > 0)
        
        total = len(payment_schedule)
        print(f"  Total invoices: {total}")
        print(f"  Early with discount: {early_with_discount} ({early_with_discount/total*100:.1f}%)")
        print(f"  Early without discount: {early_no_discount} ({early_no_discount/total*100:.1f}%)")
        print(f"  On time: {on_time} ({on_time/total*100:.1f}%)")
        print(f"  Late: {late} ({late/total*100:.1f}%)")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def run_ar_optimization(data):
    """Run the AR optimization model"""
    print("\n=== Running Accounts Receivable Optimization ===")
    
    # Configure the AR optimizer
    config = {
        "optimization_mode": "cash_flow",  # cash_flow, customer, financing, or custom
        "horizon": 90,
        "max_financing": 500000,
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
    optimizer = AROptimizer(config)
    processed_data = optimizer.preprocess_data(data)
    optimizer.build_model(processed_data)
    results = optimizer.solve()
    
    # Print summary of results
    if results["status"] == "optimal":
        print(f"Optimization successful! Objective value: {results['objective_value']:.2f}")
        print("\nKey Metrics:")
        metrics = results["key_metrics"]
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
            
        print("\nCollection Strategy Summary:")
        action_plan = results["action_plan"]
        
        # Group actions by type
        action_counts = {}
        for action in action_plan:
            action_type = action["action"]
            if action_type not in action_counts:
                action_counts[action_type] = 0
            action_counts[action_type] += 1
        
        print(f"  Total actions recommended: {len(action_plan)}")
        for action_type, count in action_counts.items():
            print(f"  {action_type}: {count}")
        
        # Show collections by timing
        ar_decisions = results["ar_decisions"]
        before_due = sum(1 for d in ar_decisions if d["collection_timing"] == "before_due")
        after_due = sum(1 for d in ar_decisions if d["collection_timing"] == "after_due")
        
        total = len(ar_decisions)
        print(f"\nExpected Collection Timing:")
        print(f"  Before due date: {before_due} ({before_due/total*100:.1f}%)")
        print(f"  After due date: {after_due} ({after_due/total*100:.1f}%)")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def run_wc_optimization(ap_data, ar_data):
    """Run the integrated Working Capital optimization model"""
    print("\n=== Running Integrated Working Capital Optimization ===")
    
    # Combine AP and AR data
    data = {
        "ap_invoices": ap_data["ap_invoices"],
        "ar_invoices": ar_data["ar_invoices"],
        "cash_position": ap_data["cash_position"],  # Use same cash position for simplicity
        "forecasts": ap_data["forecasts"]  # Use same forecasts for simplicity
    }
    
    # Configure the WC optimizer
    config = {
        "optimization_mode": "balanced",  # cash_flow, balanced, relationship, cost, or custom
        "horizon": 90,
        "max_financing": 500000,
        "financing_rate": 0.0001,  # Daily rate (approx 3.65% annually)
        # AP-specific weights
        "ap_weights": {
            "discount_weight": 0.9,
            "penalty_weight": 1.0,
            "relationship_weight": 0.5,
            "cash_weight": 0.7
        },
        # AR-specific weights
        "ar_weights": {
            "collection_weight": 0.9,
            "financing_weight": 0.7,
            "relationship_weight": 0.6,
            "transaction_weight": 0.4
        }
    }
    
    # Initialize and run the optimizer
    optimizer = WorkingCapitalOptimizer(config)
    processed_data = optimizer.preprocess_data(data)
    optimizer.build_model(processed_data)
    results = optimizer.solve()
    
    # Print summary of results
    if results["status"] == "optimal":
        print(f"Optimization successful! Objective value: {results['objective_value']:.2f}")
        print("\nKey Metrics:")
        metrics = results["key_metrics"]
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
            
        print("\nWorking Capital Management Summary:")
        print(f"  Net Working Capital: ${metrics.get('net_working_capital', 0):.2f}")
        print(f"  Cash Conversion Cycle: {metrics.get('cash_conversion_cycle', 0):.1f} days")
        print(f"  Maximum External Financing: ${metrics.get('max_financing_needed', 0):.2f}")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")
    
    return results

def generate_explanations(results):
    """Generate explanations for optimization results"""
    print("\n=== Generating AI Explanations ===")
    
    # Initialize the explanation engine
    engine = ExplanationEngine()
    
    # Generate general explanation
    print("\n1. General Explanation:")
    general_explanation = engine.generate_explanation(
        results,
        explanation_type="general"
    )
    print(general_explanation)
    
    # Generate decision explanation for a specific invoice
    if "ap_decisions" in results:
        # Find a strategic supplier's invoice
        ap_decisions = results["ap_decisions"]
        strategic_decision = next(
            (d for d in ap_decisions if d.get("supplier_id", "").startswith("Supplier_A")),
            ap_decisions[0] if ap_decisions else None
        )
        
        if strategic_decision:
            invoice_id = strategic_decision["invoice_id"]
            print(f"\n2. Decision Explanation for invoice {invoice_id}:")
            decision_explanation = engine.generate_explanation(
                results,
                explanation_type="decision",
                specific_entity=invoice_id
            )
            print(decision_explanation)
    
    # Generate root cause analysis
    print("\n3. Root Cause Analysis:")
    root_cause_explanation = engine.generate_explanation(
        results,
        explanation_type="root_cause",
        specific_question="What factors are driving our working capital needs?"
    )
    print(root_cause_explanation)

def main():
    """Main demo script"""
    print("=== Working Capital Optimization Demo ===")
    print("Generating sample data...")
    
    # Generate sample data
    ap_data = generate_sample_ap_data(20)
    ar_data = generate_sample_ar_data(20)
    
    # Run individual optimizations
    ap_results = run_ap_optimization(ap_data)
    ar_results = run_ar_optimization(ar_data)
    
    # Run integrated optimization
    wc_results = run_wc_optimization(ap_data, ar_data)
    
    # Generate explanations
    generate_explanations(wc_results)
    
    print("\nDemo complete! The working capital optimization system successfully:")
    print("1. Optimized accounts payable schedule")
    print("2. Optimized accounts receivable collection strategy")
    print("3. Integrated both for holistic working capital management")
    print("4. Generated AI explanations for the results")
    
    # Save results for further analysis
    print("\nSaving results to output directory...")
    os.makedirs("output", exist_ok=True)
    
    with open("output/ap_results.json", "w") as f:
        json.dump(ap_results, f, indent=2)
    
    with open("output/ar_results.json", "w") as f:
        json.dump(ar_results, f, indent=2)
    
    with open("output/wc_results.json", "w") as f:
        json.dump(wc_results, f, indent=2)
    
    print("Results saved to output directory.")

if __name__ == "__main__":
    main()
