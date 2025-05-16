"""
Simplified Accounts Receivable Optimizer for feasibility testing
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

def get_customer_factor(customer_segment):
    """Return customer_factor string based on customer_segment ('high', 'low', or other)."""
    if customer_segment == "high":
        return "good"
    elif customer_segment == "low":
        return "bad"
    else:
        return "average"

import pandas as pd
from ortools.linear_solver import pywraplp

from .base import BaseOptimizer


class SimpleAROptimizer(BaseOptimizer):
    """
    Simplified Accounts Receivable optimizer to ensure feasibility.
    Focuses on the core functionality without complex constraints.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simple AR optimizer.
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        super().__init__(config)
        
        # Set up time periods
        self.horizon = config.get("horizon", 90)  # Default 90-day horizon
        self.time_periods = range(1, self.horizon + 1)
        
        # Collection actions
        self.collection_actions = config.get("collection_actions", [
            "reminder",
            "call",
            "escalate",
            "discount_offer",
            "late_fee"
        ])
        
        # Initialize data structures
        self.ar_invoices = None
        self.cash_position = None

    @staticmethod
    def assign_customer_segments(ar_invoices, high_threshold=1, low_threshold=3):
        # Count due invoices per customer
        from collections import defaultdict
        due_counts = defaultdict(int)
        for invoice in ar_invoices:
            if invoice.get("days_overdue", 0) > 1:
                due_counts[invoice["business_entity_id"]] += 1

        # Assign segment based on due count
        for invoice in ar_invoices:
            due = due_counts[invoice["business_entity_id"]]
            if due >= low_threshold:
                segment = "low"
            elif due <= high_threshold:
                segment = "high"
            else:
                segment = "medium"
            invoice["customer_segment"] = segment
        return ar_invoices
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess AR data with a simplified approach.
        
        Args:
            data: Input data including AR invoices
        """
        # Extract and process invoice data
        ar_invoices = data.get("ar_invoices", [])
        ar_invoices = SimpleAROptimizer.assign_customer_segments(ar_invoices)

        # Store for later use
        self.ar_invoices = ar_invoices

        # Process collection actions, probabilities, and costs
        processed_invoices = []
        for idx, invoice in enumerate(ar_invoices):
            # Extract invoice details
            invoice_id = invoice.get("id", f"inv_{idx}")
            amount = invoice.get("amount", 0)
            due_date = invoice.get("due_date", self.horizon // 2)
            issue_date = invoice.get("issue_date", 1)
            customer_id = invoice.get("customer_id", "unknown")
            customer_segment = invoice.get("customer_segment", "medium")
            
            # Add to processed invoices
            processed_invoice = {
                "invoice_id": invoice_id,
                "amount": amount,
                "due_date": due_date,
                "issue_date": issue_date,
                "customer_id": customer_id,
                "customer_segment": customer_segment,
                "business_entity_id": invoice.get("business_entity_id", "unknown"),
                "business_entity": invoice.get("business_entity", "unknown"),
                "total_invoices": invoice.get("total_invoices", 1),
                "total_invoiced_original": invoice.get("total_invoiced_original", amount),
                "total_paid_original": invoice.get("total_paid_original", 0),
                "total_balance_original": invoice.get("total_balance_original", amount),
                "days_overdue": invoice.get("days_overdue", 0),
                "paid_amount_original": invoice.get("paid_amount_original", 0),
                "exchange_rate": invoice.get("exchange_rate", 1.0),
                "currency_code": invoice.get("currency_code"),
                "invoice_number": invoice.get("invoice_number", invoice_id),
                "total_original": invoice.get("total_original", amount)
            }
            
            # Add action costs and effectiveness based on customer segment
            action_costs = {}
            action_effectiveness = {}
            
            for action in self.collection_actions:
                # Basic action costs
                if action == "reminder":
                    action_costs[action] = 2
                elif action == "call":
                    action_costs[action] = 15
                elif action == "escalate":
                    action_costs[action] = 50
                elif action == "discount_offer":
                    action_costs[action] = amount * 0.02  # 2% discount
                else:  # late_fee
                    action_costs[action] = 0
                
                # Basic effectiveness
                if action == "reminder":
                    effectiveness = 0.1
                elif action == "call":
                    effectiveness = 0.2
                elif action == "escalate":
                    effectiveness = 0.25
                elif action == "discount_offer":
                    effectiveness = 0.3
                else:  # late_fee
                    effectiveness = 0.1
                
                # Adjust for customer segment
                if customer_segment == "high":
                    effectiveness *= 1.2
                elif customer_segment == "low":
                    effectiveness *= 0.8
                
                action_effectiveness[action] = effectiveness
            
            processed_invoice["action_costs"] = action_costs
            processed_invoice["action_effectiveness"] = action_effectiveness
            
            processed_invoices.append(processed_invoice)
        
        # Structure for optimization
        processed_data = {
            "invoices": processed_invoices,
            "time_periods": list(self.time_periods),
            "collection_actions": self.collection_actions
        }
        
        return processed_data
    
    def build_model(self, data: Dict[str, Any]) -> None:
        """
        Build a simplified AR optimization model.
        
        Args:
            data: Preprocessed AR data
        """
        self.processed_data = data
        
        # Extract components from processed data
        invoices = self.processed_data["invoices"]
        time_periods = self.processed_data["time_periods"]
        collection_actions = self.processed_data["collection_actions"]
        
        # Create decision variables
        self._create_decision_variables(invoices, time_periods, collection_actions)
        
        # Set up constraints
        self._setup_constraints(invoices, time_periods, collection_actions)
        
        # Set up objective function
        self._setup_objective_function(invoices, time_periods, collection_actions)
    
    def _create_decision_variables(self, invoices, time_periods, collection_actions):
        """Create AR decision variables"""
        # Action decision variables (binary) - whether to take action a for invoice i at time t
        self.variables["action"] = {}
        for i, invoice in enumerate(invoices):
            for action in collection_actions:
                for t in time_periods:
                    var_name = f"action_{i}_{action}_{t}"
                    self.variables["action"][(i, action, t)] = self.solver.BoolVar(var_name)
        
        # Collection amount variables (continuous) - expected amount collected for invoice i
        self.variables["collect_amount"] = {}
        for i, invoice in enumerate(invoices):
            var_name = f"collect_amount_{i}"
            self.variables["collect_amount"][i] = self.solver.NumVar(0, invoice["amount"], var_name)
        
        # Collection time variables (continuous) - expected collection day for invoice i
        self.variables["collect_time"] = {}
        for i, _ in enumerate(invoices):
            var_name = f"collect_time_{i}"
            self.variables["collect_time"][i] = self.solver.NumVar(0, max(time_periods), var_name)
    
    def _setup_constraints(self, invoices, time_periods, collection_actions):
        """Set up simplified constraints"""
        # Limit number of actions per invoice
        for i, _ in enumerate(invoices):
            # Limit total actions per invoice
            total_actions = self.solver.Sum(
                self.variables["action"][(i, action, t)]
                for action in collection_actions
                for t in time_periods
            )
            # Allow up to 3 actions per invoice
            self.solver.Add(total_actions <= 3)
            
            # Limit actions per time period
            for t in time_periods:
                actions_at_t = self.solver.Sum(
                    self.variables["action"][(i, action, t)]
                    for action in collection_actions
                )
                # At most one action per time period per invoice
                self.solver.Add(actions_at_t <= 1)
        
        # Set collection amounts based on actions
        for i, invoice in enumerate(invoices):
            # Base collection without any actions (based on due date)
            due_date = invoice["due_date"]
            base_collection_rate = 0.6  # 60% chance of collection without any actions
            
            # Total effectiveness from all actions
            total_effectiveness = self.solver.Sum(
                invoice["action_effectiveness"][action] * self.variables["action"][(i, action, t)]
                for action in collection_actions
                for t in time_periods
            )
            
            # Cap total effectiveness at 0.4 (so total collection rate maxes at 100%)
            # This uses a simple linear model: collection_rate = base_rate + effectiveness
            # We'll estimate collection amount as amount * collection_rate
            max_additional_effectiveness = 1.0 - base_collection_rate
            
            # Collection amount = amount * (base_rate + min(total_effectiveness, max_additional))
            # We'll use an upper bound constraint since we're maximizing collection
            self.solver.Add(
                self.variables["collect_amount"][i] <= 
                invoice["amount"] * (base_collection_rate + max_additional_effectiveness)
            )
            
            # Set lower bound based on actions
            # This constraint ensures actions have a meaningful impact
            self.solver.Add(
                self.variables["collect_amount"][i] >= 
                invoice["amount"] * base_collection_rate
            )
            
            # Each action contributes to earlier collection time
            # Base collection time without actions (estimated as due date + 15 days)
            base_collection_time = due_date + 15
            
            # Each action can accelerate collection by up to "effectiveness * 30 days"
            time_acceleration = self.solver.Sum(
                invoice["action_effectiveness"][action] * 30 * self.variables["action"][(i, action, t)]
                for action in collection_actions
                for t in time_periods
            )
            
            # Collection time = base_time - acceleration (with a minimum of due date)
            self.solver.Add(
                self.variables["collect_time"][i] <= base_collection_time
            )
            
            self.solver.Add(
                self.variables["collect_time"][i] >= due_date
            )
            
            # Encourage earlier collection time
            self.solver.Add(
                self.variables["collect_time"][i] <= base_collection_time - time_acceleration
            )
    
    def _setup_objective_function(self, invoices, time_periods, collection_actions):
        """Set up the objective function to maximize collected amount and minimize time"""
        objective = self.solver.Objective()
        objective.SetMaximization()
        
        # Maximize collected amount
        for i, invoice in enumerate(invoices):
            # NPV factor - earlier collections are worth more
            npv_factor = 10000  # Large enough to dominate the objective
            objective.SetCoefficient(self.variables["collect_amount"][i], npv_factor)
        
        # Minimize collection time (negative coefficient in maximization problem)
        for i, invoice in enumerate(invoices):
            # Weight by invoice amount to prioritize larger invoices
            time_weight = -1 * invoice["amount"] / 100  # Scaled to avoid dominating amount term
            objective.SetCoefficient(self.variables["collect_time"][i], time_weight)
            
        # Minimize action costs (negative coefficient in maximization problem)
        for i, invoice in enumerate(invoices):
            for action in collection_actions:
                for t in time_periods:
                    cost = invoice["action_costs"][action]
                    # Scale down cost impact to avoid dominating the objective
                    cost_weight = -0.1 * cost
                    objective.SetCoefficient(self.variables["action"][(i, action, t)], cost_weight)
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the AR optimization model.
        
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
        collection_actions = self.processed_data["collection_actions"]
        
        # Extract action plan
        action_plan = []
        for i, invoice in enumerate(invoices):
            for action in collection_actions:
                for t in time_periods:
                    if self.variables["action"][(i, action, t)].solution_value() > 0.5:  # Binary variable is 1
                        action_info = {
                            "invoice_number": invoice["invoice_number"],
                            "customer_id": invoice["customer_id"],
                            "action": action,
                            "day": t,
                            "cost": invoice["action_costs"][action],
                            "expected_effect": invoice["action_effectiveness"][action] * invoice["amount"]
                        }
                        action_plan.append(action_info)
        
        # Extract collection estimates
        collection_estimates = {}
        total_expected_collections = 0
        total_action_costs = 0
        average_collection_time = 0
        total_weight = 0
        
        for i, invoice in enumerate(invoices):
            collection_amount = self.variables["collect_amount"][i].solution_value()
            collection_time = self.variables["collect_time"][i].solution_value()
            
            collection_estimates[invoice["invoice_id"]] = {
                "business_entity_id": invoice["business_entity_id"],
                "business_entity": invoice["business_entity"],
                "total_invoices": invoice["total_invoices"],
                "total_invoiced_original": invoice["total_invoiced_original"],
                "total_paid_original": invoice["total_paid_original"],
                "total_balance_original": invoice["total_balance_original"],
                "days_overdue": invoice["days_overdue"],
                "paid_amount_original": invoice["paid_amount_original"],
                "exchange_rate": invoice["exchange_rate"],
                "currency_code": invoice["currency_code"],
                "invoice_number": invoice["invoice_number"],
                "total_original": invoice["total_original"],
                "customer_segment": invoice["customer_segment"]
            }
            
            total_expected_collections += collection_amount
            average_collection_time += collection_time * collection_amount
            total_weight += collection_amount
            
            # Calculate action costs for this invoice
            for action in collection_actions:
                for t in time_periods:
                    if self.variables["action"][(i, action, t)].solution_value() > 0.5:
                        total_action_costs += invoice["action_costs"][action]
        
        # Calculate DSO
        dso = average_collection_time / total_weight if total_weight > 0 else 0
        
        # Generate AR decisions for explanation engine
        ar_decisions = []
        for i, invoice in enumerate(invoices):
            invoice_actions = [
                action for action in action_plan
                if action["invoice_id"] == invoice["invoice_id"]
            ]
            
            collection_time = self.variables["collect_time"][i].solution_value()
            collection_amount = self.variables["collect_amount"][i].solution_value()
            
            # Assign customer_factor
            # For reusability elsewhere, see the get_customer_factor helper function at the top of this file
            if invoice["customer_segment"] == "high":
                customer_factor = "good"
            elif invoice["customer_segment"] == "low":
                customer_factor = "bad"
            else:
                customer_factor = "average"
            # Determine timing relative to due date
            if collection_time <= invoice["days_overdue"]:
                timing = "before_due"
                days_early = invoice["days_overdue"] - collection_time
                days_late = 0
                ar_decision = {
                    "business_entity_id": invoice["business_entity_id"],
                    "business_entity": invoice["business_entity"],
                    "total_invoices": invoice["total_invoices"],
                    "total_invoiced_original": invoice["total_invoiced_original"],
                    "total_paid_original": invoice["total_paid_original"],
                    "total_balance_original": invoice["total_balance_original"],
                    "days_overdue": invoice["days_overdue"],
                    "paid_amount_original": invoice["paid_amount_original"],
                    "exchange_rate": invoice["exchange_rate"],
                    "invoice_number": invoice["invoice_number"],
                    "timing": timing,
                    "days_early": days_early,
                    "days_late": days_late,
                    "total_original": invoice["total_original"],
                    "actions": [action["action"] for action in invoice_actions],
                    "action_days": [action["day"] for action in invoice_actions],
                    "total_action_cost": sum(action["cost"] for action in invoice_actions),
                    "customer_segment": invoice["customer_segment"]
                }
                ar_decisions.append(ar_decision)
            else:
                timing = "after_due"
                days_early = 0
                days_late = collection_time - invoice["days_overdue"]
                ar_decision = {
                    "business_entity_id": invoice["business_entity_id"],
                    "business_entity": invoice["business_entity"],
                    "total_invoices": invoice["total_invoices"],
                    "total_invoiced_original": invoice["total_invoiced_original"],
                    "total_paid_original": invoice["total_paid_original"],
                    "total_balance_original": invoice["total_balance_original"],
                    "days_overdue": invoice["days_overdue"],
                    "paid_amount_original": invoice["paid_amount_original"],
                    "exchange_rate": invoice["exchange_rate"],
                    "invoice_number": invoice["invoice_number"],
                    "timing": timing,
                    "days_early": days_early,
                    "days_late": days_late,
                    "total_original": invoice["total_original"],
                    "actions": [action["action"] for action in invoice_actions],
                    "action_days": [action["day"] for action in invoice_actions],
                    "total_action_cost": sum(action["cost"] for action in invoice_actions),
                    "customer_segment": invoice["customer_segment"],
                    "customer_factor": customer_factor
                }
                ar_decisions.append(ar_decision)

        # Structure results
        results = {
            "status": "optimal",
            "objective_value": self.solver.Objective().Value(),
            "action_plan": action_plan,
            "collection_estimates": collection_estimates,
            "ar_decisions": ar_decisions,
            "key_metrics": {
                "total_expected_collections": total_expected_collections,
                "total_action_costs": total_action_costs,
                "average_collection_time": dso,
                "dso": dso,
                "collection_rate": total_expected_collections / sum(invoice["amount"] for invoice in invoices) if invoices else 0
            }
        }
        
        return results
