"""
Base optimizer class for working capital optimization
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers in the working capital system.
    Defines the common interface and utility methods.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the optimizer with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.solver = None
        self.variables = {}
        self.constraints = {}
        self.results = None
        self.explanation_data = None
        
        # Initialize solver
        self._initialize_solver()
    
    def _initialize_solver(self):
        """Initialize the OR-Tools solver based on configuration"""
        solver_type = self.config.get("solver_type", "SCIP")
        
        if solver_type == "SCIP":
            self.solver = pywraplp.Solver.CreateSolver("SCIP")
        elif solver_type == "CBC":
            self.solver = pywraplp.Solver.CreateSolver("CBC")
        elif solver_type == "GLOP":
            self.solver = pywraplp.Solver.CreateSolver("GLOP")
        else:
            raise ValueError(f"Unsupported solver type: {solver_type}")
            
        if not self.solver:
            raise RuntimeError(f"Failed to create solver of type {solver_type}")
    
    @abstractmethod
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input data into format suitable for optimization.
        
        Args:
            data: Raw input data for optimization
            
        Returns:
            Preprocessed data
        """
        pass
    
    @abstractmethod
    def build_model(self, data: Dict[str, Any]) -> None:
        """
        Build the optimization model with variables, constraints, and objective.
        
        Args:
            data: Preprocessed input data
        """
        pass
    
    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        """
        Solve the optimization model.
        
        Returns:
            Dictionary containing optimization results
        """
        pass
    
    @abstractmethod
    def extract_solution(self) -> Dict[str, Any]:
        """
        Extract and structure the optimization solution.
        
        Returns:
            Structured solution data
        """
        pass
    
    def prepare_explanation_data(self) -> Dict[str, Any]:
        """
        Prepare optimization results for explanation engine.
        
        Returns:
            Dictionary with structured data for LLM explanations
        """
        if not self.results:
            raise ValueError("No optimization results available. Run solve() first.")
        
        # Structure data for explanation engine (to be implemented by subclasses)
        explanation_data = {
            "model_type": self.__class__.__name__,
            "objective_value": self.results.get("objective_value"),
            "key_metrics": self.results.get("key_metrics", {}),
            "key_decisions": self.results.get("key_decisions", []),
            "configuration": self.config,
        }
        
        self.explanation_data = explanation_data
        return explanation_data
    
    def is_feasible(self) -> bool:
        """
        Check if the optimization problem has a feasible solution.
        
        Returns:
            Boolean indicating feasibility
        """
        return self.solver.status() in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]
    
    def get_objective_value(self) -> float:
        """
        Get the objective function value.
        
        Returns:
            Objective function value
        """
        if not self.is_feasible():
            return float('nan')
        return self.solver.Objective().Value()
    
    def sensitivity_analysis(self, parameter: str, range_values: List[float]) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis by varying a parameter.
        
        Args:
            parameter: The parameter to vary
            range_values: List of values to test for the parameter
            
        Returns:
            Dictionary mapping parameter values to objective values
        """
        results = {"parameter_values": [], "objective_values": []}
        original_value = self.config.get(parameter)
        
        for value in range_values:
            # Update parameter
            self.config[parameter] = value
            
            # Rebuild and resolve model
            self.build_model(self.last_input_data)
            self.solve()
            
            # Store results
            results["parameter_values"].append(value)
            results["objective_values"].append(self.get_objective_value())
        
        # Restore original value
        self.config[parameter] = original_value
        
        return results
