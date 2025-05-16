"""
Additional methods for WorkingCapitalOptimizer class
"""

def solve(self) -> Dict[str, Any]:
    """
    Solve the working capital optimization model.
    
    Returns:
        Optimization results
    """
    status = self.solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        self.results = self.extract_solution()
        return self.results
    elif status == pywraplp.Solver.FEASIBLE:
        self.results = self.extract_solution()
        self.results["status"] = "feasible_non_optimal"
        return self.results
    else:
        return {"status": "infeasible", "error": "No feasible solution found"}

def extract_solution(self) -> Dict[str, Any]:
    """
    Extract and structure the optimization solution.
    
    Returns:
        Structured solution data
    """
    # Extract relevant data from processed data
    ap_invoices = self.processed_data["ap_invoices"]
    ar_invoices = self.processed_data["ar_invoices"]
    time_periods = self.processed_data["time_data"]["time_periods"]
    scenario = self.scenario_names[0]  # Use primary scenario for results
    
    # Extract AP decisions
    ap_decisions = []
    for i, invoice in enumerate(ap_invoices):
        for t in time_periods:
            if self.variables["ap_pay"][(i, t, scenario)].solution_value() > 0.5:  # Binary variable is 1
                # Determine payment timing
                due_date = invoice.get("due_date", 0)
                discount_date = invoice.get("discount_date")
                
                # Calculate payment info
                if discount_date is not None and t <= discount_date:
                    payment_timing = "early_with_discount"
                    days_early = due_date - t
                    days_late = 0
                elif t <= due_date:
                    payment_timing = "early_no_discount" if t < due_date else "on_time"
                    days_early = due_date - t
                    days_late = 0
                else:
                    payment_timing = "late"
                    days_early = 0
                    days_late = t - due_date
                
                # Calculate payment amount
                amount = invoice.get("amount", 0)
                if payment_timing == "early_with_discount" and "discount_rate" in invoice:
                    payment_amount = amount * (1 - invoice["discount_rate"])
                    discount_amount = amount * invoice["discount_rate"]
                elif payment_timing == "late" and "penalty_rate" in invoice:
                    penalty_amount = amount * invoice["penalty_rate"] * days_late
                    payment_amount = amount + penalty_amount
                else:
                    payment_amount = amount
                    discount_amount = 0
                    penalty_amount = 0
                
                # Create decision record
                ap_decision = {
                    "invoice_id": invoice.get("id", f"AP{i+1}"),
                    "supplier_id": invoice.get("supplier_id", f"Supplier_{i+1}"),
                    "payment_day": t,
                    "payment_timing": payment_timing,
                    "payment_amount": payment_amount,
                    "days_early": days_early,
                    "days_late": days_late,
                    "discount_captured": payment_timing == "early_with_discount",
                    "penalty_paid": payment_timing == "late"
                }
                
                if payment_timing == "early_with_discount":
                    ap_decision["discount_amount"] = discount_amount
                elif payment_timing == "late":
                    ap_decision["penalty_amount"] = penalty_amount
                
                ap_decisions.append(ap_decision)
    
    # Extract AR decisions
    ar_decisions = []
    collection_actions = self.config.get("collection_actions", 
                                       ["reminder", "call", "escalate", "discount_offer", "late_fee"])
    
    for i, invoice in enumerate(ar_invoices):
        # Find actions taken for this invoice
        invoice_actions = []
        for action in collection_actions:
            for t in time_periods:
                if self.variables["ar_action"][(i, action, t, scenario)].solution_value() > 0.5:  # Binary variable is 1
                    invoice_actions.append({
                        "action": action,
                        "day": t
                    })
        
        # Calculate expected collections
        expected_collections = {}
        total_expected = 0
        weighted_day_sum = 0
        
        for t in time_periods:
            collection_amount = self.variables["ar_collect"][(i, t, scenario)].solution_value()
            if collection_amount > 0:
                expected_collections[t] = collection_amount
                total_expected += collection_amount
                weighted_day_sum += t * collection_amount
        
        # Calculate expected collection day
        expected_collection_day = 0
        if total_expected > 0:
            expected_collection_day = weighted_day_sum / total_expected
            
        # Determine collection timing
        due_date = invoice.get("due_date", 0)
        if expected_collection_day <= due_date:
            collection_timing = "before_due"
        else:
            collection_timing = "after_due"
            
        # Create decision record
        ar_decision = {
            "invoice_id": invoice.get("id", f"AR{i+1}"),
            "customer_id": invoice.get("customer_id", f"Customer_{i+1}"),
            "expected_collection_day": expected_collection_day,
            "expected_amount": total_expected,
            "collection_timing": collection_timing,
            "actions": [a["action"] for a in invoice_actions],
            "action_days": [a["day"] for a in invoice_actions],
            "customer_segment": invoice.get("customer_segment", "medium")
        }
        
        ar_decisions.append(ar_decision)
    
    # Extract cash flow
    cash_flow = {}
    borrowing = {}
    for t in time_periods:
        cash_flow[t] = self.variables["cash"][(t, scenario)].solution_value()
        borrowing[t] = self.variables["borrow"][(t, scenario)].solution_value()
    
    # Calculate key metrics
    # AP metrics
    total_ap_paid = sum(decision["payment_amount"] for decision in ap_decisions)
    total_discounts = sum(
        decision.get("discount_amount", 0) for decision in ap_decisions 
        if decision["discount_captured"]
    )
    total_penalties = sum(
        decision.get("penalty_amount", 0) for decision in ap_decisions 
        if decision["penalty_paid"]
    )
    
    # Average days to payment (DPO)
    ap_weighted_days = sum(
        (decision["days_early"] - decision["days_late"]) * decision["payment_amount"]
        for decision in ap_decisions
    )
    dpo = -ap_weighted_days / total_ap_paid if total_ap_paid > 0 else 0
    
    # AR metrics
    total_ar_expected = sum(decision["expected_amount"] for decision in ar_decisions)
    
    # Calculate DSO
    ar_weighted_days = sum(
        decision["expected_collection_day"] * decision["expected_amount"]
        for decision in ar_decisions
    )
    dso = ar_weighted_days / total_ar_expected if total_ar_expected > 0 else 0
    
    # Calculate cash conversion cycle
    cash_conversion_cycle = dso + dpo
    
    # Calculate maximum financing needed
    max_financing = max(borrowing.values())
    
    # Calculate net working capital
    # Simplified - would need more asset/liability data for complete calculation
    net_working_capital = max(0, cash_flow[max(time_periods)])
    
    # Combine results
    results = {
        "status": "optimal",
        "objective_value": self.solver.Objective().Value(),
        "ap_decisions": ap_decisions,
        "ar_decisions": ar_decisions,
        "cash_flow": cash_flow,
        "borrowing": borrowing,
        "key_metrics": {
            "total_ap_paid": total_ap_paid,
            "total_discounts_captured": total_discounts,
            "total_penalties_paid": total_penalties,
            "total_ar_expected": total_ar_expected,
            "dpo": dpo,
            "dso": dso,
            "cash_conversion_cycle": cash_conversion_cycle,
            "max_financing_needed": max_financing,
            "net_working_capital": net_working_capital,
            "financing_costs": sum(borrowing.values()) * self.config.get("financing_rate", 0.0001)
        },
        "optimization_mode": self.config.get("optimization_mode", "balanced")
    }
    
    return results
