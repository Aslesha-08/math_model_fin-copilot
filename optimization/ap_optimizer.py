"""
Accounts Payable Optimizer - Implementation of the mathematical model for AP optimization
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from .base import BaseOptimizer


class APOptimizer(BaseOptimizer):
    """
    Accounts Payable optimizer implementation.
    Determines optimal payment timing for supplier invoices to balance
    cost minimization, supplier satisfaction, and cash flow.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AP optimizer.
        
        Args:
            config: Configuration dictionary with AP optimization parameters
        """
        super().__init__(config)
        
        # Initialize objective weights based on chosen mode
        self.optimization_mode = config.get("optimization_mode", "cost")
        
        if self.optimization_mode == "cost":
            # Cost minimization mode
            self.discount_weight = 1.0
            self.penalty_weight = 1.0
            self.relationship_weight = 0.2
            self.cash_weight = 0.5
        elif self.optimization_mode == "supplier":
            # Supplier satisfaction mode
            self.discount_weight = 0.7
            self.penalty_weight = 1.0
            self.relationship_weight = 1.0
            self.cash_weight = 0.3
        elif self.optimization_mode == "cash":
            # Cash flow optimization mode
            self.discount_weight = 0.6
            self.penalty_weight = 0.7
            self.relationship_weight = 0.4
            self.cash_weight = 1.0
        else:
            # Custom weights
            self.discount_weight = config.get("discount_weight", 0.8)
            self.penalty_weight = config.get("penalty_weight", 0.9)
            self.relationship_weight = config.get("relationship_weight", 0.6)
            self.cash_weight = config.get("cash_weight", 0.7)
            
        # Set up time periods
        self.horizon = config.get("horizon", 90)  # Default 90-day horizon
        self.time_periods = range(1, self.horizon + 1)
        
        # Initialize data structures
        self.ap_invoices = None
        self.cash_position = None
        self.forecasts = None
        
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess AP data for optimization.
        
        Args:
            data: Input data including AP invoices, cash position, forecasts
            
        Returns:
            Preprocessed data
        """
        # Extract invoice data
        ap_invoices = data.get("ap_invoices", [])
        cash_position = data.get("cash_position", {"initial_balance": 0})
        forecasts = data.get("forecasts", {})
        
        # Store for later use
        self.ap_invoices = ap_invoices
        self.cash_position = cash_position
        self.forecasts = forecasts
        
        # Calculate payment periods and costs
        processed_invoices = []
        for idx, invoice in enumerate(ap_invoices):
            # Extract invoice details
            invoice_id = invoice.get("id", f"inv_{idx}")
            amount = invoice.get("amount", 0)
            due_date = 0  # Force all due dates in the past so all payments are late
            discount_date = invoice.get("discount_date")
            discount_rate = invoice.get("discount_rate", 0)
            penalty_rate = invoice.get("penalty_rate", 0)
            supplier_id = invoice.get("supplier_id", "unknown")
            supplier_priority = invoice.get("supplier_priority", "medium")
            
            # Calculate payment costs for each time period
            payment_costs = {}
            for t in self.time_periods:
                if discount_date is not None and t <= discount_date:
                    # Early payment discount
                    payment_costs[t] = amount * (1 - discount_rate)
                elif t <= due_date:
                    # On time payment (full amount)
                    payment_costs[t] = amount
                else:
                    # Late payment (with penalty)
                    days_late = t - due_date
                    penalty = amount * penalty_rate * days_late
                    payment_costs[t] = amount + penalty
            
            # Calculate relationship impact for each time period
            relationship_impact = {}
            for t in self.time_periods:
                if discount_date is not None and t <= discount_date:
                    # Very early payment (best for relationship)
                    relationship_impact[t] = 1.0
                elif t <= due_date:
                    # On time payment (good for relationship)
                    relationship_impact[t] = 0.8
                else:
                    # Late payment (bad for relationship)
                    days_late = t - due_date
                    # Exponential decay of relationship score with late days
                    relationship_impact[t] = max(0, 0.8 * np.exp(-0.05 * days_late))
            
            # Add to processed invoices
            processed_invoices.append({
                "invoice_id": invoice_id,
                "amount": amount,
                "due_date": due_date,
                "discount_date": discount_date,
                "discount_rate": discount_rate,
                "penalty_rate": penalty_rate,
                "supplier_id": supplier_id,
                "supplier_priority": supplier_priority,
                "payment_costs": payment_costs,
                "relationship_impact": relationship_impact
            })
        
        # Preprocess cash flow data
        cash_flow = {
            "initial_balance": cash_position.get("initial_balance", 0),
            "inflows": {},
            "outflows": {}
        }
        
        for t in self.time_periods:
            forecast = forecasts.get(t, {})
            cash_flow["inflows"][t] = forecast.get("inflow", 0)
            cash_flow["outflows"][t] = forecast.get("outflow", 0)
        
        # Structure for optimization
        processed_data = {
            "invoices": processed_invoices,
            "cash_flow": cash_flow,
            "time_periods": list(self.time_periods)
        }
        
        self.last_input_data = data
        
        return processed_data
    
    def build_model(self, data: Dict[str, Any]) -> None:
        """
        Build the AP optimization model.
        
        Args:
            data: Preprocessed AP data
        """
        # Use data directly if provided, otherwise use last processed data
        if data:
            self.processed_data = data
        
        # Extract components from processed data
        invoices = self.processed_data["invoices"]
        cash_flow = self.processed_data["cash_flow"]
        time_periods = self.processed_data["time_periods"]
        
        # Create decision variables
        self._create_ap_decision_variables(invoices, time_periods)
        
        # Set up cash flow constraints
        self._setup_ap_cash_flow_constraints(invoices, cash_flow, time_periods)
        
        # Set up invoice payment constraints (each invoice paid exactly once)
        self._setup_ap_invoice_constraints(invoices, time_periods)
        
        # Set up objective function based on mode
        self._setup_ap_objective_function(invoices, time_periods)
    
    def _create_ap_decision_variables(self, invoices, time_periods):
        """Create AP decision variables"""
        # Payment decision variables (binary)
        self.variables["pay"] = {}
        for i, invoice in enumerate(invoices):
            for t in time_periods:
                var_name = f"pay_{i}_{t}"
                self.variables["pay"][(i, t)] = self.solver.BoolVar(var_name)
        
        # Cash balance variables
        self.variables["cash"] = {}
        for t in time_periods:
            var_name = f"cash_{t}"
            self.variables["cash"][t] = self.solver.NumVar(0, self.solver.infinity(), var_name)
        
        # Borrowing variables
        self.variables["borrow"] = {}
        for t in time_periods:
            var_name = f"borrow_{t}"
            max_borrow = self.config.get("max_borrowing", self.solver.infinity())
            self.variables["borrow"][t] = self.solver.NumVar(0, max_borrow, var_name)
        
        # Minimum cash variable (for cash flow optimization)
        self.variables["min_cash"] = self.solver.NumVar(0, 1000000, "min_cash")  # Set max to 1,000,000 as a reasonable upper bound
    
    def _setup_ap_invoice_constraints(self, invoices, time_periods):
        """Each invoice must be paid exactly once"""
        for i, invoice in enumerate(invoices):
            constraint = self.solver.Constraint(1, 1)  # Exactly one payment date per invoice
            for t in time_periods:
                constraint.SetCoefficient(self.variables["pay"][(i, t)], 1)
    
    def _setup_ap_cash_flow_constraints(self, invoices, cash_flow, time_periods):
        """Set up cash flow balance constraints"""
        # Initial cash balance
        initial_cash = cash_flow["initial_balance"]
        
        # First day cash balance
        t = time_periods[0]
        constraint = self.solver.Constraint(0, 0)  # Cash_1 = Initial + Inflows - Outflows - Payments + Borrowing
        constraint.SetCoefficient(self.variables["cash"][t], 1)
        
        # Add inflows and outflows
        inflow = cash_flow["inflows"].get(t, 0)
        outflow = cash_flow["outflows"].get(t, 0)
        constant = initial_cash + inflow - outflow
        
        # Add payments
        for i, invoice in enumerate(invoices):
            payment_cost = invoice["payment_costs"].get(t, invoice["amount"])
            constraint.SetCoefficient(self.variables["pay"][(i, t)], -payment_cost)
        
        # Add borrowing
        constraint.SetCoefficient(self.variables["borrow"][t], 1)
        
        # Set constant term
        constraint.SetBounds(constant, constant)
        
        # Minimum cash balance constraints
        for t in time_periods:
            constraint = self.solver.Constraint(0, self.solver.infinity())
            constraint.SetCoefficient(self.variables["min_cash"], 1)
            constraint.SetCoefficient(self.variables["cash"][t], -1)
        
        # Cash balance for subsequent days
        for t_idx in range(1, len(time_periods)):
            t = time_periods[t_idx]
            prev_t = time_periods[t_idx - 1]
            
            constraint = self.solver.Constraint(0, 0)  # Cash_t = Cash_{t-1} + Inflows - Outflows - Payments + Borrowing
            constraint.SetCoefficient(self.variables["cash"][t], 1)
            constraint.SetCoefficient(self.variables["cash"][prev_t], -1)
            
            # Add inflows and outflows
            inflow = cash_flow["inflows"].get(t, 0)
            outflow = cash_flow["outflows"].get(t, 0)
            constant = inflow - outflow
            
            # Add payments
            for i, invoice in enumerate(invoices):
                payment_cost = invoice["payment_costs"].get(t, invoice["amount"])
                constraint.SetCoefficient(self.variables["pay"][(i, t)], -payment_cost)
            
            # Add borrowing
            constraint.SetCoefficient(self.variables["borrow"][t], 1)
            
            # Set constant term
            constraint.SetBounds(constant, constant)
        
        # Maximum cash balance variable and constraints
        self.variables["max_cash"] = self.solver.NumVar(0, self.solver.infinity(), "max_cash")
        for t in time_periods:
            constraint = self.solver.Constraint(0, self.solver.infinity())
            constraint.SetCoefficient(self.variables["max_cash"], 1)
            constraint.SetCoefficient(self.variables["cash"][t], -1)
    
    def _setup_ap_objective_function(self, invoices, time_periods):
        """Set up the objective function based on optimization mode"""
        objective = self.solver.Objective()
        objective.SetMinimization()
        
        # Payment cost component (discount capture, penalty avoidance)
        for i, invoice in enumerate(invoices):
            for t in time_periods:
                payment_cost = invoice["payment_costs"].get(t, invoice["amount"])
                # Apply supplier priority weighting
                priority_factor = 1.0
                if invoice["supplier_priority"] == "high":
                    priority_factor = 0.8  # Discount cost for high priority suppliers
                elif invoice["supplier_priority"] == "low":
                    priority_factor = 1.2  # Increase cost for low priority suppliers
                
                coefficient = self.discount_weight * payment_cost * priority_factor
                objective.SetCoefficient(self.variables["pay"][(i, t)], coefficient)
        
        # Supplier relationship component
        # This is applied as a penalty for late payments
        if self.optimization_mode in ["supplier", "custom"]:
            for i, invoice in enumerate(invoices):
                for t in time_periods:
                    relationship_impact = invoice["relationship_impact"].get(t, 0)
                    
                    # Convert to cost (1 - impact) * factor to make it a minimization
                    relationship_cost = (1 - relationship_impact) * invoice["amount"]
                    
                    # Apply supplier priority weighting
                    priority_factor = 1.0
                    if invoice["supplier_priority"] == "high":
                        priority_factor = 3.0  # Triple impact for strategic suppliers
                    elif invoice["supplier_priority"] == "medium":
                        priority_factor = 1.5  # 50% more impact for medium priority
                    
                    coefficient = self.relationship_weight * relationship_cost * priority_factor
                    objective.SetCoefficient(self.variables["pay"][(i, t)], coefficient)
        
        # Cash flow component (borrowing cost)
        borrowing_rate = self.config.get("borrowing_rate", 0.0001)  # Daily rate, roughly 3.65% per year
        for t in time_periods:
            coefficient = self.cash_weight * borrowing_rate
            objective.SetCoefficient(self.variables["borrow"][t], coefficient)
        
        # Cash flow component (minimum cash maximization)
        # When in cash flow mode, we want to maximize minimum cash, which is the same as minimizing negative of min cash
        if self.optimization_mode == "cash":
            objective.SetCoefficient(self.variables["min_cash"], -1 * self.cash_weight)
    
    def calculate_priority(self, item: Dict[str, Any]) -> float:
        """
        Calculate priority score for a payment schedule item.
        
        Args:
            item: Payment schedule item with days_overdue and total_balance_original
            
        Returns:
            Priority score (0-1)
        """
        # Normalize days overdue to a scale of 0-1 (assuming max overdue days is 30)
        normalized_days = min(item.get('days_overdue', 0) / 30, 1.0)
        # Normalize balance to a scale of 0-1 (using log to reduce impact of very large balances)
        normalized_balance = min(1.0, item.get('total_balance_original', 0) / 10000000)  # Normalize to millions
        
        # Combine both factors with weights (adjust weights as needed)
        priority_score = (0.7 * normalized_days + 0.3 * normalized_balance)
        return priority_score

    def solve(self) -> Dict[str, Any]:
        """
        Solve the AP optimization model.
        
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
        invoices = self.processed_data["invoices"]
        time_periods = self.processed_data["time_periods"]
        
        # Extract payment decisions
        payment_schedule = []
        for i, invoice in enumerate(invoices):
            for t in time_periods:
                if self.variables["pay"][(i, t)].solution_value() > 0.5:
                    payment_info = {
                        "invoice_id": invoice["invoice_id"],
                        "supplier_id": invoice["supplier_id"],
                        "amount": invoice["amount"],
                        "payment_day": t,
                        "payment_amount": invoice["payment_costs"][t],
                        "discount_captured": False,
                        "penalty_paid": False,
                        "days_early": 0,
                        "days_late": 0,
                        "invoice_index": i  # Store the index for reference
                    }
                    # Calculate early/late days and discount/penalty
                    if invoice["discount_date"] is not None and t <= invoice["discount_date"]:
                        payment_info["discount_captured"] = True
                        payment_info["days_early"] = invoice["due_date"] - t
                    elif t <= invoice["due_date"]:
                        payment_info["days_early"] = invoice["due_date"] - t
                    else:
                        payment_info["days_late"] = max(1, t - invoice["due_date"])
                        payment_info["penalty_paid"] = True
                    payment_schedule.append(payment_info)

        # After all calculations, create a stripped version for output
        payment_schedule_stripped = []
        for payment in payment_schedule:
            i = payment["invoice_index"]
            entity_info = self.ap_invoices[i] if hasattr(self, 'ap_invoices') and self.ap_invoices else {}
            payment_stripped = {
                "business_entity_id": entity_info.get("business_entity_id"),
                "business_entity": entity_info.get("business_entity"),
                "total_invoices": entity_info.get("total_invoices"),
                "total_invoiced_original": entity_info.get("total_invoiced_original"),
                "total_paid_original": entity_info.get("total_paid_original"),
                "total_balance_original": entity_info.get("total_balance_original"),
                "days_overdue": entity_info.get("days_overdue"),
                "exchange_rate": entity_info.get("exchange_rate"),
                "currency_code": entity_info.get("currency_code"),
            }
            payment_schedule_stripped.append(payment_stripped)

        
        # Extract cash flow
        cash_flow = {}
        borrowing = {}
        for t in time_periods:
            cash_flow[t] = self.variables["cash"][t].solution_value()
            borrowing[t] = self.variables["borrow"][t].solution_value()
        
        # Calculate key metrics
        total_paid = sum(payment["payment_amount"] for payment in payment_schedule)
        total_discounts = sum(
            payment["amount"] * invoices[i]["discount_rate"] 
            for i, payment in enumerate(payment_schedule) 
            if payment["discount_captured"]
        )
        total_penalties = sum(
            payment["amount"] * invoices[i]["penalty_rate"] * payment["days_late"]
            for i, payment in enumerate(payment_schedule) 
            if payment["penalty_paid"]
        )
        total_borrowing = sum(borrowing.values())
        average_days_to_payment = sum(
            payment["days_early"] - payment["days_late"] 
            for payment in payment_schedule
        ) / len(payment_schedule) if payment_schedule else 0
        
        # Add priorities to payment schedule items
        for item in payment_schedule:
            item['priority'] = self.calculate_priority(item)
        
        # Sort payment schedule by priority (highest first)
        payment_schedule = sorted(payment_schedule, key=lambda x: x.get('priority', 0), reverse=True)
        
        # Add priorities to AP decisions
        for item in payment_schedule:
            item['priority'] = self.calculate_priority(item)
        
        # Structure results
        results = {
            "status": "optimal",
            "objective_value": self.solver.Objective().Value(),
            "payment_schedule": payment_schedule_stripped,
            "cash_flow": cash_flow,
            "borrowing": borrowing,
            "key_metrics": {
                "total_paid": total_paid,
                "total_discounts_captured": total_discounts,
                "total_penalties_paid": total_penalties,
                "total_borrowing": total_borrowing,
                "minimum_cash_balance": self.variables["min_cash"].solution_value() if "min_cash" in self.variables else min(cash_flow.values()),
                "average_days_to_payment": average_days_to_payment,
                "dpo": -average_days_to_payment  # DPO is negative of average days to payment
            }
        }
        
        # Generate AP decisions list for the explanation engine
        ap_decisions = []
        for payment in payment_schedule:
            invoice_index = payment["invoice_index"]
            entity_info = self.ap_invoices[invoice_index] if hasattr(self, 'ap_invoices') and self.ap_invoices else {}
            # Determine payment timing category
            if payment["discount_captured"]:
                timing = "early_with_discount"
            elif payment["days_early"] > 0:
                timing = "early_no_discount"
            elif payment["days_late"] == 0:
                timing = "on_time"
            else:
                timing = "late"
            # Calculate payment impact
            payment_impact = payment["payment_amount"]
            if timing == "early_with_discount" and "discount_rate" in entity_info:
                impact_description = f"Saved ${payment['amount'] * entity_info.get('discount_rate', 0):.2f} with early payment discount"
            elif timing == "late" and "penalty_rate" in entity_info:
                impact_description = f"Incurred ${payment['amount'] * entity_info.get('penalty_rate', 0) * payment['days_late']:.2f} in late payment penalties"
            else:
                impact_description = "Payment not yet made, Overdue"
            ap_decisions.append({
                "business_entity_id": entity_info.get("business_entity_id"),
                "business_entity": entity_info.get("business_entity"),
                "total_invoices": entity_info.get("total_invoices"),
                "total_invoiced_original": entity_info.get("total_invoiced_original"),
                "total_paid_original": entity_info.get("total_paid_original"),
                "total_balance_original": entity_info.get("total_balance_original"),
                "currency_code": entity_info.get("currency_code"),
                "days_overdue": entity_info.get("days_overdue"),
                "description": impact_description
            })
        
        results["ap_decisions"] = ap_decisions
        
        return results
