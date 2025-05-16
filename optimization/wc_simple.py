"""
Simplified Working Capital Optimizer that coordinates AP and AR optimization
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from .base import BaseOptimizer
from .ap_optimizer import APOptimizer
from .ar_simple import SimpleAROptimizer


class SimpleWorkingCapitalOptimizer(BaseOptimizer):
    """
    Simplified Working Capital optimizer that coordinates AP and AR optimization.
    This approach runs both optimizers and then combines their results into
    an integrated view of working capital management.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the working capital optimizer.
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        super().__init__(config)
        
        # Store configuration
        self.config = config
        
        # Set up sub-optimizers
        self.ap_config = self._prepare_ap_config(config)
        self.ar_config = self._prepare_ar_config(config)
        
        self.ap_optimizer = APOptimizer(self.ap_config)
        self.ar_optimizer = SimpleAROptimizer(self.ar_config)
        
        # Initialize results storage
        self.ap_results = None
        self.ar_results = None
        self.processed_data = None
    
    def _prepare_ap_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for AP optimizer"""
        ap_config = {
            "optimization_mode": config.get("ap_optimization_mode", "balanced"),
            "horizon": config.get("horizon", 90),
            "max_borrowing": config.get("max_financing", 500000),
            "borrowing_rate": config.get("financing_rate", 0.0001)
        }
        
        # Add AP-specific weights if provided
        if "ap_weights" in config:
            ap_config.update(config["ap_weights"])
        
        return ap_config
    
    def _prepare_ar_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for AR optimizer"""
        ar_config = {
            "optimization_mode": config.get("ar_optimization_mode", "balanced"),
            "horizon": config.get("horizon", 90),
            "max_financing": config.get("max_financing", 500000),
            "financing_rate": config.get("financing_rate", 0.0001),
            "collection_actions": config.get("collection_actions", [
                "reminder", "call", "escalate", "discount_offer", "late_fee"
            ])
        }
        
        # Add AR-specific weights if provided
        if "ar_weights" in config:
            ar_config.update(config["ar_weights"])
        
        return ar_config
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input data for working capital optimization.
        Splits the data and prepares it for AP and AR optimizers.
        
        Args:
            data: Raw input data including AP and AR invoices
            
        Returns:
            Preprocessed data
        """
        # Extract data components
        ap_invoices = data.get("ap_invoices", [])
        ar_invoices = data.get("ar_invoices", [])
        cash_position = data.get("cash_position", {})
        forecasts = data.get("forecasts", {})
        
        # Prepare AP data
        ap_data = {
            "ap_invoices": ap_invoices,
            "cash_position": cash_position,
            "forecasts": forecasts
        }
        
        # Prepare AR data
        ar_data = {
            "ar_invoices": ar_invoices,
            "cash_position": cash_position,
            "forecasts": forecasts
        }
        
        # Store data for later use
        self.processed_data = {
            "ap_data": ap_data,
            "ar_data": ar_data,
            "combined_data": data
        }
        
        return self.processed_data
    
    def build_model(self, data: Dict[str, Any]) -> None:
        """
        Build the integrated working capital optimization model.
        This implementation simply delegates to the sub-optimizers.
        
        Args:
            data: Preprocessed input data
        """
        # Use data directly if provided, otherwise use last processed data
        if data:
            self.processed_data = data
        
        # No actual model building needed as we'll run sub-optimizers separately
        # This is a placeholder to comply with the abstract method requirement
        pass
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the working capital optimization problem.
        This runs both AP and AR optimizers and combines their results.
        
        Returns:
            Optimization results
        """
        # Get AP optimization results
        ap_data = self.processed_data["ap_data"]
        ap_processed_data = self.ap_optimizer.preprocess_data(ap_data)
        self.ap_optimizer.build_model(ap_processed_data)
        self.ap_results = self.ap_optimizer.solve()
        
        # Get AR optimization results
        ar_data = self.processed_data["ar_data"]
        ar_processed_data = self.ar_optimizer.preprocess_data(ar_data)
        self.ar_optimizer.build_model(ar_processed_data)
        self.ar_results = self.ar_optimizer.solve()
        
        # Check if either optimization failed
        if self.ap_results.get("status") not in ["optimal", "feasible_non_optimal"]:
            return {"status": "infeasible", "error": "AP optimization failed"}
        
        if self.ar_results.get("status") not in ["optimal", "feasible_non_optimal"]:
            return {"status": "infeasible", "error": "AR optimization failed"}
        
        # Extract and combine the results
        results = self.extract_solution()
        
        return results
    
    def extract_solution(self) -> Dict[str, Any]:
        """
        Extract and structure the optimization solution.
        Combines AP and AR results into an integrated working capital view.
        
        Returns:
            Structured solution data
        """
        # Extract key metrics from AP results
        ap_metrics = self.ap_results.get("key_metrics", {})
        ap_decisions = self.ap_results.get("ap_decisions", [])
        
        # Extract key metrics from AR results
        ar_metrics = self.ar_results.get("key_metrics", {})
        ar_decisions = self.ar_results.get("ar_decisions", [])
        
        # Extract cash flows
        ap_cash_flow = self.ap_results.get("cash_flow", {})
        
        # Calculate working capital metrics
        dpo = ap_metrics.get("dpo", 0)
        dso = ar_metrics.get("dso", 0)
        
        # Cash Conversion Cycle = DSO + DIO - DPO
        # Since we don't model inventory, we'll use DSO - DPO
        cash_conversion_cycle = dso - dpo
        
        # Calculate total outgoing cash flow (accounts payable)
        total_payables = ap_metrics.get("total_paid", 0)
        
        # Calculate total incoming cash flow (accounts receivable)
        total_receivables = ar_metrics.get("total_expected_collections", 0)
        
        # Calculate net working capital
        net_working_capital = total_receivables - total_payables
        
        # Calculate maximum financing needed
        total_borrowing = sum(ap_metrics.get("total_borrowing", 0) for t in ap_cash_flow)
        
        # Calculate financing costs
        financing_rate = self.config.get("financing_rate", 0.0001)
        financing_costs = total_borrowing * financing_rate
        
        # Combine into working capital metrics
        wc_metrics = {
            "cash_conversion_cycle": cash_conversion_cycle,
            "dso": dso,
            "dpo": dpo,
            "net_working_capital": net_working_capital,
            "total_payables": total_payables,
            "total_receivables": total_receivables,
            "max_financing_needed": total_borrowing,
            "financing_costs": financing_costs,
            
            # Include key metrics from AP and AR
            "ap_discounts_captured": ap_metrics.get("total_discounts_captured", 0),
            "ap_penalties_paid": ap_metrics.get("total_penalties_paid", 0),
            "ar_collection_rate": ar_metrics.get("collection_rate", 0),
            "ar_action_costs": ar_metrics.get("total_action_costs", 0)
        }
        
        # Combine results
        results = {
            "status": "optimal",
            "objective_value": self.ap_results.get("objective_value", 0) + self.ar_results.get("objective_value", 0),
            "ap_decisions": ap_decisions,
            "ar_decisions": ar_decisions,
            "key_metrics": wc_metrics,
            "optimization_mode": self.config.get("optimization_mode", "balanced")
        }
        
        return results
