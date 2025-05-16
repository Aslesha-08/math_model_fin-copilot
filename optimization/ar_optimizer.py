"""
Accounts Receivable Optimizer - Implementation of the probability-based model for AR optimization
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from .base import BaseOptimizer


class AROptimizer(BaseOptimizer):
    """
    Accounts Receivable optimizer implementation.
    Determines optimal collection strategies for customer invoices using
    a probability-based model to balance cash flow acceleration,
    customer relationships, and collection costs.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AR optimizer.
        
        Args:
            config: Configuration dictionary with AR optimization parameters
        """
        super().__init__(config)
        
        # Initialize objective weights based on chosen mode
        self.optimization_mode = config.get("optimization_mode", "cash_flow")
        
        if self.optimization_mode == "cash_flow":
            # Cash flow maximization mode
            self.collection_weight = 1.0
            self.financing_weight = 0.8
            self.relationship_weight = 0.4
            self.transaction_weight = 0.3
        elif self.optimization_mode == "customer":
            # Customer relationship mode
            self.collection_weight = 0.6
            self.financing_weight = 0.5
            self.relationship_weight = 1.0
            self.transaction_weight = 0.4
        elif self.optimization_mode == "financing":
            # Financing need minimization mode
            self.collection_weight = 0.7
            self.financing_weight = 1.0
            self.relationship_weight = 0.3
            self.transaction_weight = 0.5
        else:
            # Custom weights
            self.collection_weight = config.get("collection_weight", 0.8)
            self.financing_weight = config.get("financing_weight", 0.7)
            self.relationship_weight = config.get("relationship_weight", 0.6)
            self.transaction_weight = config.get("transaction_weight", 0.4)
            
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
        self.forecasts = None
        
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess AR data for optimization.
        
        Args:
            data: Input data including AR invoices, cash position, forecasts
            
        Returns:
            Preprocessed data
        """
        # Extract components from input data with defaults
        invoices = data.get("ar_invoices", [])
        self.cash_position = data.get("cash_position", {"initial_balance": 0.0})
        self.forecasts = data.get("forecasts", {})
        
        # Ensure cash position is properly initialized
        if not isinstance(self.cash_position, dict):
            self.cash_position = {"initial_balance": float(self.cash_position) if self.cash_position else 0.0}
        
        # Ensure initial_balance is a float
        self.cash_position["initial_balance"] = float(self.cash_position.get("initial_balance", 0.0))
        
        # Process invoices
        processed_invoices = []
        total_invoice_amount = 0.0
        
        # Calculate total invoice amount for scaling
        for invoice in invoices:
            try:
                amount = float(invoice.get("amount", 0.0))
                total_invoice_amount += amount
            except (ValueError, TypeError):
                continue
        
        # If no invoices, return empty result
        if not invoices or total_invoice_amount <= 0:
            return {
                "invoices": [],
                "cash_flow": {"initial_balance": self.cash_position["initial_balance"], "inflows": {}, "outflows": {}},
                "time_periods": list(self.time_periods),
                "collection_actions": self.collection_actions,
                "financing_rate": self.config.get("financing_rate", 0.00005)
            }
        
        # Process each invoice
        for i, invoice in enumerate(invoices):
            try:
                # Extract basic invoice information with validation
                invoice_id = str(invoice.get("invoice_number", f"inv_{i}"))
                
                # Handle amount with validation
                try:
                    amount = float(invoice.get("amount", 0.0))
                    if amount <= 0:
                        print(f"Warning: Invoice {invoice_id} has zero/negative amount, skipping")
                        continue
                except (ValueError, TypeError):
                    print(f"Warning: Invalid amount for invoice {invoice_id}, skipping")
                    continue
                    
                # Preserve business entity information
                business_entity = invoice.get("business_entity", "")
                business_entity_id = invoice.get("business_entity_id", "")
                total_invoices = invoice.get("total_invoices", 0)
                total_invoiced_original = invoice.get("total_invoiced_original", 0)
                total_paid_original = invoice.get("total_paid_original", 0)
                total_balance_original = invoice.get("total_balance_original", 0)
                days_overdue = invoice.get("days_overdue", 0)
                paid_amount_original = invoice.get("paid_amount_original", 0)
                exchange_rate = invoice.get("exchange_rate", 1.0)
                invoice_number = invoice.get("invoice_number", invoice_id)
                total_original = invoice.get("total_original", amount)
                currency_code = invoice.get("currency_code", "USD")
                
                # Handle due date with validation
                try:
                    due_date = int(invoice.get("days_overdue", 0)) + 1  # Convert to days from now
                    due_date = max(1, min(due_date, self.horizon))  # Ensure within horizon
                except (ValueError, TypeError):
                    due_date = self.horizon  # Default to end of horizon if invalid
                
                # Handle issue date with validation
                try:
                    issue_date = int(invoice.get("issue_date", 1))
                    issue_date = max(1, min(issue_date, self.horizon))  # Ensure within horizon
                except (ValueError, TypeError):
                    issue_date = 1  # Default to day 1 if invalid
                
                # Ensure issue date is before due date
                if issue_date >= due_date:
                    issue_date = max(1, due_date - 1)
                
                # Handle payment probability with validation
                try:
                    payment_probability = float(invoice.get("payment_probability", 0.8))
                    payment_probability = max(0.0, min(1.0, payment_probability))  # Clamp to [0,1]
                except (ValueError, TypeError):
                    payment_probability = 0.8  # Default if invalid
                
                # Customer information
                customer_id = str(invoice.get("customer_id", f"cust_{i}"))
                customer_segment = str(invoice.get("customer_segment", "medium")).lower()
                if customer_segment not in ["high", "medium", "low"]:
                    customer_segment = "medium"  # Default to medium if invalid
                
                # Calculate base payment probabilities for each time period without any action
                base_probabilities = {}
            except Exception as e:
                print(f"Error processing invoice {i}: {str(e)}")
                continue
                
            # Calculate base payment probabilities for each time period without any action
            base_probabilities = {}
            for t in self.time_periods:
                # Model increasing likelihood of payment as due date approaches and after
                if t < issue_date:
                    base_probabilities[t] = 0.0  # Can't be paid before issuance
                elif t < due_date:
                    # Sigmoid function that rises as we approach due date
                    days_to_due = due_date - t
                    days_since_issue = t - issue_date
                    total_lead_time = due_date - issue_date
                    
                    if total_lead_time > 0:
                        progress = days_since_issue / total_lead_time
                        base_probabilities[t] = payment_probability * (1 / (1 + np.exp(-10 * (progress - 0.7))))
                    else:
                        base_probabilities[t] = 0.05
                elif t == due_date:
                    base_probabilities[t] = payment_probability * 0.3  # Spike on due date
                else:
                    # Decaying probability after due date
                    days_late = t - due_date
                    base_probabilities[t] = payment_probability * 0.3 * np.exp(-0.05 * days_late)
            
            # Calculate collection action effects
            action_effects = {}
            for action in self.collection_actions:
                action_effects[action] = {}
                
                for t in self.time_periods:
                    # Define action effectiveness based on timing and customer segment
                    effect = 0.0
                    
                    if action == "reminder":
                        # Reminders most effective before due date
                        if t < due_date:
                            days_to_due = due_date - t
                            if days_to_due <= 7:  # Effective 7 days before due date
                                effect = 0.15
                            elif days_to_due <= 14:
                                effect = 0.10
                            else:
                                effect = 0.05
                                
                    elif action == "call":
                        # Calls effective around and after due date
                        if t >= due_date - 3:
                            days_from_due = abs(t - due_date)
                            effect = 0.20 * np.exp(-0.1 * days_from_due)
                            
                    elif action == "escalate":
                        # Escalation effective only after due date
                        if t > due_date:
                            days_late = t - due_date
                            if days_late >= 14:  # Only escalate after 14 days late
                                effect = 0.25
                                
                    elif action == "discount_offer":
                        # Discount offers most effective before due date
                        if t < due_date:
                            effect = 0.30  # Strong effect
                        else:
                            effect = 0.15  # Less effective after due date
                            
                    elif action == "late_fee":
                        # Late fees only applicable after due date
                        if t > due_date:
                            effect = 0.10
                    
                    # Adjust effect based on customer segment
                    if customer_segment == "high":
                        effect *= 1.2  # More responsive
                    elif customer_segment == "low":
                        effect *= 0.8  # Less responsive
                        
                    action_effects[action][t] = effect
            
            # Calculate collection costs
            action_costs = {
                "reminder": 2,    # Email/SMS cost
                "call": 15,       # Human call time cost
                "escalate": 50,   # Management involvement cost
                "discount_offer": amount * 0.02,  # 2% discount
                "late_fee": 0     # No direct cost to implement
            }
            
            # Calculate relationship impact
            relationship_impact = {
                "reminder": -0.05,      # Slight negative
                "call": -0.15,          # Moderate negative
                "escalate": -0.40,      # Significant negative
                "discount_offer": 0.10, # Positive
                "late_fee": -0.25       # Negative
            }
            
            # Adjust relationship impact based on customer segment
            if customer_segment == "high":
                # Strategic customers - more sensitive to negative actions
                for action in relationship_impact:
                    if relationship_impact[action] < 0:
                        relationship_impact[action] *= 1.5
            
            # Add to processed invoices
            processed_invoices.append({
                "invoice_id": invoice_id,
                "amount": amount,
                "due_date": due_date,
                "issue_date": issue_date,
                "customer_id": customer_id,
                "customer_segment": customer_segment,
                "base_probabilities": base_probabilities,
                "action_effects": action_effects,
                "action_costs": action_costs,
                "relationship_impact": relationship_impact,
                
                # Store business entity information
                "business_entity": business_entity,
                "business_entity_id": business_entity_id,
                "total_invoices": total_invoices,
                "total_invoiced_original": total_invoiced_original,
                "total_paid_original": total_paid_original,
                "total_balance_original": total_balance_original,
                "days_overdue": days_overdue,
                "paid_amount_original": paid_amount_original,
                "exchange_rate": exchange_rate,
                "invoice_number": invoice_number,
                "total_original": total_original,
                "currency_code": currency_code
            })
        
        # Preprocess cash flow data
        cash_flow = {
            "initial_balance": self.cash_position.get("initial_balance", 0),
            "inflows": {},
            "outflows": {}
        }
        
        for t in self.time_periods:
            forecast = self.forecasts.get(t, {})
            cash_flow["inflows"][t] = forecast.get("inflow", 0)
            cash_flow["outflows"][t] = forecast.get("outflow", 0)
        
        # Structure for optimization
        processed_data = {
            "invoices": processed_invoices,
            "cash_flow": cash_flow,
            "time_periods": list(self.time_periods),
            "collection_actions": self.collection_actions,
            "financing_rate": self.config.get("financing_rate", 0.0001)  # Daily rate
        }
        
        self.last_input_data = data
        
        return processed_data
    
    def build_model(self, data: Dict[str, Any]) -> None:
        """
        Build the AR optimization model.
        
        Args:
            data: Preprocessed AR data
        """
        # Use data directly if provided, otherwise use last processed data
        if data:
            self.processed_data = data
        
        # Extract components from processed data
        invoices = self.processed_data["invoices"]
        cash_flow = self.processed_data["cash_flow"]
        time_periods = self.processed_data["time_periods"]
        collection_actions = self.processed_data["collection_actions"]
        
        # Create decision variables
        self._create_ar_decision_variables(invoices, time_periods, collection_actions)
        
        # Set up cash flow constraints
        self._setup_ar_cash_flow_constraints(invoices, cash_flow, time_periods, collection_actions)
        
        # Set up collection action constraints
        self._setup_ar_action_constraints(invoices, time_periods, collection_actions)
        
        # Set up objective function based on mode
        self._setup_ar_objective_function(invoices, time_periods, collection_actions)
    
    def _create_ar_decision_variables(self, invoices, time_periods, collection_actions):
        """Create AR decision variables"""
        # Collection action decision variables (binary)
        self.variables["action"] = {}
        for i, invoice in enumerate(invoices):
            for action in collection_actions:
                for t in time_periods:
                    var_name = f"action_{i}_{action}_{t}"
                    self.variables["action"][(i, action, t)] = self.solver.BoolVar(var_name)
        
        # Expected collection variables (continuous)
        self.variables["collection"] = {}
        for i, invoice in enumerate(invoices):
            for t in time_periods:
                var_name = f"collection_{i}_{t}"
                self.variables["collection"][(i, t)] = self.solver.NumVar(0, invoice["amount"], var_name)
        
        # Cash balance variables
        self.variables["cash"] = {}
        for t in time_periods:
            var_name = f"cash_{t}"
            self.variables["cash"][t] = self.solver.NumVar(0, self.solver.infinity(), var_name)
        
        # Borrowing/financing variables
        self.variables["financing"] = {}
        for t in time_periods:
            var_name = f"financing_{t}"
            max_financing = self.config.get("max_financing", self.solver.infinity())
            self.variables["financing"][t] = self.solver.NumVar(0, max_financing, var_name)
        
        # Customer relationship variables (continuous, represents impact score)
        self.variables["relationship"] = {}
        for i, invoice in enumerate(invoices):
            var_name = f"relationship_{i}"
            # Relationship ranges from -1 (worst) to 1 (best)
            self.variables["relationship"][i] = self.solver.NumVar(-1, 1, var_name)
    
    def _setup_ar_action_constraints(self, invoices, time_periods, collection_actions):
        """Set up constraints on collection actions"""
        # Limit frequency of actions for each invoice
        for i, invoice in enumerate(invoices):
            for action in collection_actions:
                # Maximum number of times each action can be taken per invoice
                max_actions = 1  # Default most actions to once per invoice
                if action == "reminder":
                    max_actions = 3  # Allow multiple reminders
                
                constraint = self.solver.Constraint(0, max_actions)
                for t in time_periods:
                    constraint.SetCoefficient(self.variables["action"][(i, action, t)], 1)
                    
            # Prevent actions too close together by adding a more reasonable constraint
            # Instead of checking every day, we'll add constraints for specific time windows
            time_windows = [
                (1, 30),   # First month
                (31, 60),  # Second month
                (61, 90)   # Third month
            ]
            
            for start_t, end_t in time_windows:
                if start_t > max(time_periods) or end_t < min(time_periods):
                    continue  # Skip if window is outside our time horizon
                    
                # Limit total actions in each window
                actual_start = max(start_t, min(time_periods))
                actual_end = min(end_t, max(time_periods))
                
                total_actions_in_window = self.solver.Sum(
                    self.variables["action"][(i, action, t)]
                    for action in collection_actions
                    for t in range(actual_start, actual_end + 1)
                    if t in time_periods
                )
                
                # Allow a reasonable number of actions in each month
                # More flexible than the previous constraint
                max_actions_per_window = 3
                self.solver.Add(total_actions_in_window <= max_actions_per_window)
        
        # Set relationship score based on actions taken
        for i, invoice in enumerate(invoices):
            relationship_sum = 0
            
            # Base relationship score (no actions = neutral)
            base_score = 0
            
            # Add impact of each action
            for action in collection_actions:
                for t in time_periods:
                    impact = invoice["relationship_impact"].get(action, 0)
                    relationship_sum += impact * self.variables["action"][(i, action, t)]
            
            # Set relationship variable
            self.solver.Add(self.variables["relationship"][i] == base_score + relationship_sum)
    
    def _setup_ar_cash_flow_constraints(self, invoices, cash_flow, time_periods, collection_actions):
        """Set up cash flow and collection probability constraints"""
        # Get configuration parameters with defaults
        min_prob = self.config.get("min_collection_probability", 0.0)
        max_prob = self.config.get("max_collection_probability", 1.0)
        allow_partial = self.config.get("allow_partial_payments", True)
        
        # Set expected collections based on probabilities and actions
        for i, invoice in enumerate(invoices):
            for t in time_periods:
                # Base probability without actions
                base_prob = invoice["base_probabilities"].get(t, 0)
                amount = invoice["amount"]
                
                # Calculate min and max collection amounts based on configuration
                min_collection = max(min_prob * amount, 0)  # At least min_prob of amount
                max_collection = min(max_prob * amount, amount)  # At most max_prob of amount or full amount
                
                # For each possible action, calculate max potential effect
                max_effect = 0
                for action in collection_actions:
                    # Consider only actions with significant effect at this time
                    max_action_effect = invoice["action_effects"][action].get(t, 0) * amount
                    max_effect += max_action_effect
                
                # Upper bound includes possible action effects
                max_collection = min(max_collection + max_effect, amount)
                
                # If partial payments are allowed, we can collect any amount up to max_collection
                # If not, we must collect either 0 or the full amount
                collection_var = self.variables["collection"][(i, t)]
                
                if allow_partial:
                    # Allow any amount between min and max collection
                    self.solver.Add(collection_var >= min_collection)
                    self.solver.Add(collection_var <= max_collection)
                else:
                    # Use binary variable to enforce either 0 or full collection
                    action_var = self.solver.BoolVar(f"collect_binary_{i}_{t}")
                    self.solver.Add(collection_var <= max_collection * action_var)
                    self.solver.Add(collection_var >= min_collection * action_var)
                    
                    # If we have a minimum collection amount, ensure at least that much is collected
                    if min_collection > 0:
                        self.solver.Add(collection_var >= min_collection * action_var)
                    
                    # Ensure we don't collect more than the invoice amount
                    self.solver.Add(collection_var <= amount)
        
        # Cash flow balance constraints
        initial_cash = cash_flow["initial_balance"]
        
        # First day cash balance
        t = time_periods[0]
        constraint = self.solver.Constraint(0, 0)  # Cash_1 = Initial + Inflows - Outflows + Collections - Action_Costs + Financing
        constraint.SetCoefficient(self.variables["cash"][t], 1)
        
        # Add inflows and outflows
        inflow = cash_flow["inflows"].get(t, 0)
        outflow = cash_flow["outflows"].get(t, 0)
        constant = initial_cash + inflow - outflow
        
        # Add collections
        for i, invoice in enumerate(invoices):
            constraint.SetCoefficient(self.variables["collection"][(i, t)], 1)
        
        # Add action costs
        for i, invoice in enumerate(invoices):
            for action in collection_actions:
                cost = invoice["action_costs"].get(action, 0)
                constraint.SetCoefficient(self.variables["action"][(i, action, t)], -cost)
        
        # Add financing
        constraint.SetCoefficient(self.variables["financing"][t], 1)
        
        # Set constant term
        constraint.SetBounds(constant, constant)
        
        # Cash balance for subsequent days
        for t_idx in range(1, len(time_periods)):
            t = time_periods[t_idx]
            prev_t = time_periods[t_idx - 1]
            
            constraint = self.solver.Constraint(0, 0)  # Cash_t = Cash_{t-1} + Inflows - Outflows + Collections - Action_Costs + Financing
            constraint.SetCoefficient(self.variables["cash"][t], 1)
            constraint.SetCoefficient(self.variables["cash"][prev_t], -1)
            
            # Add inflows and outflows
            inflow = cash_flow["inflows"].get(t, 0)
            outflow = cash_flow["outflows"].get(t, 0)
            constant = inflow - outflow
            
            # Add collections
            for i, invoice in enumerate(invoices):
                constraint.SetCoefficient(self.variables["collection"][(i, t)], 1)
            
            # Add action costs
            for i, invoice in enumerate(invoices):
                for action in collection_actions:
                    cost = invoice["action_costs"].get(action, 0)
                    constraint.SetCoefficient(self.variables["action"][(i, action, t)], -cost)
            
            # Add financing
            constraint.SetCoefficient(self.variables["financing"][t], 1)
            
            # Set constant term
            constraint.SetBounds(constant, constant)
    
    def _setup_ar_objective_function(self, invoices, time_periods, collection_actions):
        """Set up the objective function based on optimization mode"""
        objective = self.solver.Objective()
        objective.SetMaximization()  # AR is primarily a maximization problem
        
        # Get configuration parameters with defaults
        financing_rate = self.processed_data.get("financing_rate", 0.00005)
        
        # 1. Collection acceleration component (NPV of collections)
        # Weighted more heavily to prioritize collections
        collection_scale = 100.0  # Scale up to ensure collections are prioritized
        for i, invoice in enumerate(invoices):
            for t in time_periods:
                # Apply time value of money - earlier collections are more valuable
                npv_factor = 1 / ((1 + financing_rate) ** t)
                # Scale by invoice amount to prioritize larger invoices
                amount_factor = invoice["amount"] / max(1, sum(inv["amount"] for inv in invoices))
                coefficient = self.collection_weight * npv_factor * amount_factor * collection_scale
                objective.SetCoefficient(self.variables["collection"][(i, t)], coefficient)
        
        # 2. Financing cost component (minimize)
        # Weighted to balance with collections but not dominate
        financing_scale = 1.0
        for t in time_periods:
            # Scale financing cost by time - more expensive to finance for longer periods
            time_factor = 1.0 + (t / len(time_periods))  # More weight on later periods
            coefficient = -self.financing_weight * financing_rate * time_factor * financing_scale
            objective.SetCoefficient(self.variables["financing"][t], coefficient)
        
        # 3. Transaction cost component (minimize)
        # Weighted to be less important than collections
        transaction_scale = 0.1
        for i, invoice in enumerate(invoices):
            for action in collection_actions:
                for t in time_periods:
                    cost = invoice["action_costs"].get(action, 0)
                    # Scale cost by invoice amount - cheaper actions for larger invoices
                    cost_factor = cost / max(1, invoice["amount"])
                    coefficient = -self.transaction_weight * cost_factor * transaction_scale
                    objective.SetCoefficient(self.variables["action"][(i, action, t)], coefficient)
        
        # 4. Relationship component (maximize)
        # Only include if in customer-focused mode or custom weights
        if self.relationship_weight > 0:
            relationship_scale = 0.5  # Balance with other objectives
            for i, invoice in enumerate(invoices):
                # Higher weight for high-value customers
                customer_factor = 1.0
                if invoice.get("customer_segment") == "high":
                    customer_factor = 3.0
                elif invoice.get("customer_segment") == "medium":
                    customer_factor = 1.5
                
                # Scale by invoice amount to prioritize relationships with larger customers
                amount_factor = invoice["amount"] / max(1, sum(inv["amount"] for inv in invoices))
                coefficient = self.relationship_weight * customer_factor * amount_factor * relationship_scale
                objective.SetCoefficient(self.variables["relationship"][i], coefficient)
        
        # 5. Penalize late payments (encourage earlier collections)
        late_payment_scale = 0.3
        for i, invoice in enumerate(invoices):
            due_date = invoice.get("due_date", max(time_periods) + 1)
            for t in time_periods:
                if t > due_date:
                    # Penalize collections that happen after the due date
                    days_late = t - due_date
                    penalty = days_late * late_payment_scale
                    objective.SetCoefficient(self.variables["collection"][(i, t)], -penalty)
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the AR optimization model.
        
        Returns:
            Optimization results
        """
        import time
        from datetime import datetime
        
        # Enable solver logging for debugging
        self.solver.EnableOutput()
        
        # Set a time limit to avoid long runs
        self.solver.SetTimeLimit(60000)  # 60 seconds
        
        # Print model statistics before solving
        print("\n=== Model Statistics ===")
        print(f"Number of variables: {self.solver.NumVariables()}")
        print(f"Number of constraints: {self.solver.NumConstraints()}")
        
        # Print configuration
        print("\n=== Configuration ===")
        print(f"Optimization mode: {self.optimization_mode}")
        print(f"Horizon: {self.horizon} days")
        print(f"Collection weight: {self.collection_weight}")
        print(f"Financing weight: {self.financing_weight}")
        print(f"Relationship weight: {self.relationship_weight}")
        print(f"Transaction weight: {self.transaction_weight}")
        
        # Print data summary
        print("\n=== Data Summary ===")
        print(f"Number of invoices: {len(self.processed_data['invoices'])}")
        print(f"Time periods: {len(self.processed_data['time_periods'])}")
        print(f"Collection actions: {self.processed_data['collection_actions']}")
        
        # Print cash flow information
        cash_flow = self.processed_data['cash_flow']
        print("\n=== Cash Flow ===")
        print(f"Initial balance: {cash_flow.get('initial_balance', 0):.2f}")
        print(f"Max financing: {self.processed_data.get('max_financing', 0):.2f}")
        
        # Print invoice summary
        total_invoice_amount = sum(inv['amount'] for inv in self.processed_data['invoices'])
        avg_invoice_amount = total_invoice_amount / max(1, len(self.processed_data['invoices']))
        print(f"\n=== Invoices ===")
        print(f"Total invoice amount: {total_invoice_amount:.2f}")
        print(f"Number of invoices: {len(self.processed_data['invoices'])}")
        print(f"Average invoice amount: {avg_invoice_amount:.2f}")
        
        # Print first few invoices for verification
        print("\n=== Sample Invoices ===")
        for i, invoice in enumerate(self.processed_data['invoices'][:3]):  # First 3 invoices
            print(f"- Invoice {i+1}:")
            print(f"  Amount: {invoice['amount']:.2f}")
            print(f"  Due in: {invoice['due_date']} days")
            print(f"  Customer: {invoice.get('customer_id', 'N/A')}")
            print(f"  Segment: {invoice.get('customer_segment', 'N/A')}")
            
            # Print base probabilities for key dates
            due_date = invoice.get('due_date', 30)  # Default to 30 days if not specified
            key_dates = [1, due_date//2, due_date, due_date + 7, due_date + 30]
            key_dates = [min(max(1, d), self.horizon) for d in key_dates if d <= self.horizon]
            
            print("  Base probabilities at key dates:")
            for t in sorted(set(key_dates)):
                prob = invoice['base_probabilities'].get(t, 0)
                print(f"    Day {t}: {prob*100:.1f}%")
        
        # Print action costs for the first invoice
        if self.processed_data['invoices']:
            first_invoice = self.processed_data['invoices'][0]
            print("\n=== Action Costs (First Invoice) ===")
            for action in self.processed_data['collection_actions']:
                cost = first_invoice['action_costs'].get(action, 0)
                print(f"- {action}: {cost:.2f}")
        
        # Export model to file for inspection
        debug_file = "ar_model_debug.txt"
        print(f"\nExporting model to {debug_file}...")
        try:
            with open(debug_file, "w") as f:
                f.write("=== AR OPTIMIZER DEBUG INFO ===\n\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Solver: {self.solver.SolverVersion() if hasattr(self.solver, 'SolverVersion') else 'Unknown'}\n")
                f.write(f"Variables: {self.solver.NumVariables()}\n")
                f.write(f"Constraints: {self.solver.NumConstraints()}\n\n")
                
                # Write configuration
                f.write("=== Configuration ===\n")
                config = {
                    'optimization_mode': self.optimization_mode,
                    'horizon': self.horizon,
                    'collection_weight': self.collection_weight,
                    'financing_weight': self.financing_weight,
                    'relationship_weight': self.relationship_weight,
                    'transaction_weight': self.transaction_weight,
                    'max_financing': self.processed_data.get('max_financing'),
                    'financing_rate': self.processed_data.get('financing_rate')
                }
                for k, v in config.items():
                    f.write(f"{k}: {v}\n")
                
                # Write variable information
                f.write("\n=== Variables (sample) ===\n")
                for var_name, var_dict in self.variables.items():
                    if isinstance(var_dict, dict):
                        f.write(f"{var_name} ({len(var_dict)}): {next(iter(var_dict.items()))[1].name() if var_dict else 'empty'}...\n")
                    else:
                        f.write(f"{var_name}: {var_dict.name()}\n")
                
                # Write constraint information
                f.write("\n=== Constraints (sample) ===\n")
                for i in range(min(10, self.solver.NumConstraints())):
                    c = self.solver.constraints()[i]
                    f.write(f"{c.name() if hasattr(c, 'name') else f'Constraint_{i}'}: {c.lb()} <= ... <= {c.ub()}\n")
                
                # Write invoice summary
                f.write("\n=== Invoice Summary ===\n")
                f.write(f"Total invoice amount: {total_invoice_amount:.2f}\n")
                f.write(f"Number of invoices: {len(self.processed_data['invoices'])}\n")
                f.write(f"Average invoice amount: {avg_invoice_amount:.2f}\n")
                
                # Write first invoice details
                if self.processed_data['invoices']:
                    inv = self.processed_data['invoices'][0]
                    f.write("\n=== First Invoice Details ===\n")
                    f.write(f"Amount: {inv['amount']:.2f}\n")
                    f.write(f"Due date: {inv.get('due_date', 'N/A')}\n")
                    f.write(f"Customer: {inv.get('customer_id', 'N/A')}\n")
                    f.write("Base probabilities:\n")
                    for t, prob in sorted(inv['base_probabilities'].items())[:5]:
                        f.write(f"  Day {t}: {prob*100:.1f}%\n")
                    if len(inv['base_probabilities']) > 5:
                        f.write(f"  ... and {len(inv['base_probabilities']) - 5} more\n")
            
            print(f"Debug information written to {debug_file}")
        except Exception as e:
            print(f"Warning: Could not write debug information: {e}")
        
        # Solve the model
        print("\n=== Solving Model ===")
        start_time = time.time()
        status = self.solver.Solve()
        solve_time = time.time() - start_time
        
        # Check solution status
        if status == pywraplp.Solver.OPTIMAL:
            print(f"\n✅ Optimal solution found in {solve_time:.2f} seconds")
            print(f"Objective value: {self.solver.Objective().Value():.2f}")
            self.results = self.extract_solution()
            self.results["status"] = "optimal"
            return self.results
            
        elif status == pywraplp.Solver.FEASIBLE:
            print(f"\n⚠️ Feasible (non-optimal) solution found in {solve_time:.2f} seconds")
            print(f"Objective value: {self.solver.Objective().Value():.2f}")
            self.results = self.extract_solution()
            self.results["status"] = "feasible_non_optimal"
            return self.results
            
        else:
            # Map status codes to names
            status_names = {
                0: 'OPTIMAL',
                1: 'FEASIBLE',
                2: 'INFEASIBLE',
                3: 'UNBOUNDED',
                4: 'ABNORMAL',
                5: 'MODEL_INVALID',
                6: 'NOT_SOLVED'
            }
            
            # Get solver statistics
            stats = {
                'wall_time': self.solver.wall_time() if hasattr(self.solver, 'wall_time') else 0,
                'iterations': self.solver.iterations() if hasattr(self.solver, 'iterations') else 0,
                'nodes': self.solver.nodes() if hasattr(self.solver, 'nodes') else 0
            }
            
            # Provide detailed error information
            error_info = {
                "status": "infeasible" if status == 2 else "failed",
                "error": "No feasible solution found" if status == 2 else "Optimization failed",
                "solver_status": status,
                "solver_status_name": status_names.get(status, f"UNKNOWN_STATUS_{status}"),
                "solve_time_seconds": solve_time,
                "num_variables": self.solver.NumVariables(),
                "num_constraints": self.solver.NumConstraints(),
                **stats
            }
            
            print("\n❌ Optimization Failed")
            print("=" * 50)
            print(f"Status: {error_info['solver_status_name']} ({error_info['solver_status']})")
            print(f"Solve time: {solve_time:.2f} seconds")
            print(f"Variables: {error_info['num_variables']}")
            print(f"Constraints: {error_info['num_constraints']}")
            
            # Problem analysis
            print("\nProblem Analysis:")
            print("-" * 30)
            
            # Cash flow analysis
            initial_cash = self.processed_data['cash_flow']['initial_balance']
            max_financing = self.processed_data.get('max_financing', 0)
            total_available = initial_cash + max_financing
            total_invoices = sum(inv['amount'] for inv in self.processed_data['invoices'])
            
            print(f"Available funds:")
            print(f"- Initial cash: {initial_cash:,.2f}")
            print(f"- Max financing: {max_financing:,.2f}")
            print(f"- Total available: {total_available:,.2f}")
            print(f"\nTotal invoice amount: {total_invoices:,.2f}")
            
            # Check if the problem is obviously infeasible
            if total_available < 0.01:  # No money available
                print("\n❌ Problem: No funds available (initial cash + financing <= 0)")
                error_info["infeasibility_reason"] = "No funds available"
            elif total_available < total_invoices * 0.1:  # Less than 10% of invoices can be paid
                print(f"\n⚠️ Warning: Available funds ({total_available:,.2f}) are very low compared to total invoices ({total_invoices:,.2f})")
                error_info["infeasibility_reason"] = "Insufficient funds"
            
            # Check for very large invoices
            max_invoice = max((inv['amount'] for inv in self.processed_data['invoices']), default=0)
            if max_invoice > total_available:
                print(f"\n⚠️ Warning: Largest invoice ({max_invoice:,.2f}) exceeds available funds ({total_available:,.2f})")
                error_info["infeasibility_reason"] = "Invoice exceeds available funds"
            
            # Check action costs
            if self.processed_data['invoices']:
                first_invoice = self.processed_data['invoices'][0]
                action_costs = {}
                for action in self.processed_data['collection_actions']:
                    cost = first_invoice['action_costs'].get(action, 0)
                    if cost > 0:
                        action_costs[action] = cost
                
                if action_costs:
                    print("\nAction costs (first invoice):")
                    for action, cost in action_costs.items():
                        print(f"- {action}: {cost:.2f}")
            
            return error_info
    
    def extract_solution(self) -> Dict[str, Any]:
        """
        Extract and structure the optimization solution.
        
        Returns:
            Structured solution data
        """
        invoices = self.processed_data["invoices"]
        time_periods = self.processed_data["time_periods"]
        collection_actions = self.processed_data["collection_actions"]
        
        # Initialize results dictionary
        results = {
            "status": "optimal",
            "objective_value": self.solver.Objective().Value(),
            "key_metrics": {},
            "action_plan": [],
            "ar_decisions": [],
            "collection_timing": {"before_due": 0, "after_due": 0}
        }
        
        # Extract collection actions
        action_plan = []
        for i, invoice in enumerate(invoices):
            invoice_actions = []
            for action in collection_actions:
                for t in time_periods:
                    if self.variables["action"][(i, action, t)].solution_value() > 0.5:  # Binary variable is 1
                        action_info = {
                            "invoice_id": invoice.get("invoice_id", f"inv_{i}"),
                            "action": action,
                            "day": t,
                            "cost": invoice["action_costs"].get(action, 0),
                            "expected_effect": invoice["action_effects"][action].get(t, 0) * invoice["amount"],
                            "relationship_impact": invoice["relationship_impact"].get(action, 0)
                        }
                        invoice_actions.append(action_info)
            
            # Sort actions by day
            invoice_actions.sort(key=lambda x: x["day"])
            action_plan.extend(invoice_actions)
        
        # Store action plan in results
        results["action_plan"] = action_plan
        
        # Calculate total collections and timing
        total_collections = 0
        before_due = 0
        after_due = 0
        
        for i, invoice in enumerate(invoices):
            due_date = invoice.get("due_date", 0)
            for t in time_periods:
                collection_amount = self.variables["collection"][(i, t)].solution_value()
                total_collections += collection_amount
                
                if t <= due_date:
                    before_due += collection_amount
                else:
                    after_due += collection_amount
        
        # Calculate key metrics
        total_invoice_amount = sum(invoice["amount"] for invoice in invoices)
        collection_rate = total_collections / total_invoice_amount if total_invoice_amount > 0 else 0
        
        # Calculate average days to collection
        weighted_days = 0
        for i, invoice in enumerate(invoices):
            for t in time_periods:
                collection_amount = self.variables["collection"][(i, t)].solution_value()
                weighted_days += t * collection_amount
        
        avg_days_to_collection = weighted_days / total_collections if total_collections > 0 else 0
        
        # Calculate financing needs
        max_financing = 0
        for t in time_periods:
            financing_amount = self.variables["financing"][t].solution_value()
            max_financing = max(max_financing, financing_amount)
        
        # Update results with calculated metrics
        results["key_metrics"] = {
            "total_collections": total_collections,
            "collection_rate": collection_rate,
            "avg_days_to_collection": avg_days_to_collection,
            "max_financing": max_financing
        }
        
        results["collection_timing"] = {
            "before_due": before_due,
            "after_due": after_due
        }
        
        # Helper function to assign customer factor
        def assign_customer_factor(customer_segment):
            if customer_segment == "high":
                return "good"
            elif customer_segment == "low":
                return "bad"
            else:
                return "average"
                
        # Generate AR decisions list
        ar_decisions = []
        for i, invoice in enumerate(invoices):
            invoice_id = invoice.get("invoice_id", f"inv_{i}")
            customer_id = invoice.get("customer_id", f"cust_{i}")
            customer_segment = invoice.get("customer_segment", "medium")
            customer_factor = assign_customer_factor(customer_segment)
            
            # Find actions for this invoice
            invoice_actions = [
                action for action in action_plan
                if action["invoice_id"] == invoice_id
            ]
            
            # Create decision record
            ar_decision = {
                "invoice_id": invoice_id,
                "amount": invoice["amount"],
                "due_date": invoice.get("due_date", 0),
                "customer_id": customer_id,
                "customer_segment": customer_segment,
                "customer_factor": customer_factor,
                
                # Include original fields from the processed invoice
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
                "total_original": invoice["total_original"],
                "currency_code": invoice.get("currency_code", "USD"),
                
                # Add collection actions
                "actions": invoice_actions
            }
            
            ar_decisions.append(ar_decision)
        
        results["ar_decisions"] = ar_decisions
        
        return results
