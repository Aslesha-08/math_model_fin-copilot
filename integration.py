"""
Integration layer connecting the optimization engine with the AI explanation system
"""
from typing import Dict, List, Any, Optional
import json
import os
import logging

from optimization.wc_optimizer import WorkingCapitalOptimizer
from optimization.ap_optimizer import APOptimizer
from optimization.ar_optimizer import AROptimizer
from ai.explanation import ExplanationEngine


class WorkingCapitalSystem:
    """
    Main integration class that coordinates the optimization engines and AI explanation system.
    This serves as the primary entry point for the working capital optimization functionality.
    """
    def __init__(self, config_path: str = None):
        """
        Initialize the working capital system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.wc_optimizer = WorkingCapitalOptimizer(self.config.get("wc_optimization", {}))
        self.ap_optimizer = APOptimizer(self.config.get("ap_optimization", {}))
        self.ar_optimizer = AROptimizer(self.config.get("ar_optimization", {}))
        self.explanation_engine = ExplanationEngine(self.config.get("ai_explanation", {}))
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("WorkingCapitalSystem")
        
        # Track optimization results
        self.last_results = None
        self.last_explanations = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "wc_optimization": {
                "alpha": 0.4,  # Liquidity weight
                "beta": 0.3,   # Financing cost weight
                "gamma": 0.2,  # Transaction cost weight
                "theta": 0.1,  # Relationship weight
                "horizon": 90,
                "scenarios": ["baseline"],
                "scenario_probs": {"baseline": 1.0},
                "solver_type": "SCIP",
                "max_borrowing": 1000000
            },
            "ap_optimization": {
                "solver_type": "SCIP"
            },
            "ar_optimization": {
                "solver_type": "SCIP",
                "collection_actions": ["reminder", "call", "escalate"]
            },
            "ai_explanation": {
                "model": "gpt-4",
                "temperature": 0.3,
                "max_tokens": 1000
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                return default_config
        else:
            return default_config
    
    def optimize_working_capital(self, data: Dict[str, Any], 
                              mode: str = "integrated") -> Dict[str, Any]:
        """
        Run the working capital optimization.
        
        Args:
            data: Input data including AP invoices, AR invoices, cash position, and forecasts
            mode: Optimization mode ("integrated", "ap_only", "ar_only")
            
        Returns:
            Optimization results
        """
        self.logger.info(f"Starting working capital optimization in {mode} mode")
        
        try:
            # Select optimization mode
            if mode == "integrated":
                results = self._run_integrated_optimization(data)
            elif mode == "ap_only":
                results = self._run_ap_optimization(data)
            elif mode == "ar_only":
                results = self._run_ar_optimization(data)
            else:
                raise ValueError(f"Unknown optimization mode: {mode}")
            
            # Store results
            self.last_results = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}
    
    def _run_integrated_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the integrated working capital optimization"""
        # Preprocess data
        processed_data = self.wc_optimizer.preprocess_data(data)
        
        # Build and solve model
        self.wc_optimizer.build_model(processed_data)
        self.wc_optimizer.solve()
        
        # Extract solution
        results = self.wc_optimizer.extract_solution()
        
        return results
    
    def _run_ap_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run accounts payable optimization only"""
        # Extract AP-specific data
        ap_data = {
            "ap_invoices": data.get("ap_invoices", []),
            "cash_position": data.get("cash_position", {}),
            "forecasts": data.get("forecasts", {})
        }
        
        # Preprocess data
        processed_data = self.ap_optimizer.preprocess_data(ap_data)
        
        # Build and solve model
        self.ap_optimizer.build_model(processed_data)
        self.ap_optimizer.solve()
        
        # Extract solution
        results = self.ap_optimizer.extract_solution()
        
        return results
    
    def _run_ar_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run accounts receivable optimization only"""
        # Extract AR-specific data
        ar_data = {
            "ar_invoices": data.get("ar_invoices", []),
            "cash_position": data.get("cash_position", {}),
            "forecasts": data.get("forecasts", {})
        }
        
        # Preprocess data
        processed_data = self.ar_optimizer.preprocess_data(ar_data)
        
        # Build and solve model
        self.ar_optimizer.build_model(processed_data)
        self.ar_optimizer.solve()
        
        # Extract solution
        results = self.ar_optimizer.extract_solution()
        
        return results
    
    def explain_results(self, explanation_type: str = "general", 
                      specific_item: str = None) -> Dict[str, Any]:
        """
        Generate explanations for optimization results using AI.
        
        Args:
            explanation_type: Type of explanation ("general", "decision", "rca")
            specific_item: ID of specific item to explain (for decision or RCA)
            
        Returns:
            Structured explanation
        """
        if not self.last_results:
            return {"error": "No optimization results available to explain"}
        
        try:
            # Prepare data for explanation
            if explanation_type == "general":
                explanation_data = self._prepare_general_explanation_data()
            elif explanation_type == "decision" and specific_item:
                explanation_data = self._prepare_decision_explanation_data(specific_item)
            elif explanation_type == "rca" and specific_item:
                explanation_data = self._prepare_rca_explanation_data(specific_item)
            else:
                return {"error": f"Invalid explanation parameters: type={explanation_type}, item={specific_item}"}
            
            # Generate explanation
            explanation = self.explanation_engine.generate_explanation(
                explanation_data, explanation_type
            )
            
            # Store explanation
            self.last_explanations = explanation
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}
    
    def _prepare_general_explanation_data(self) -> Dict[str, Any]:
        """Prepare data for general explanation"""
        # Extract objective weights from config
        objective_weights = {
            "liquidity": self.config["wc_optimization"]["alpha"],
            "financing": self.config["wc_optimization"]["beta"],
            "transaction": self.config["wc_optimization"]["gamma"],
            "relationship": self.config["wc_optimization"]["theta"]
        }
        
        # Get key metrics
        key_metrics = self.last_results.get("key_metrics", {})
        
        # Identify key decisions
        key_decisions = self._extract_key_decisions()
        
        # Identify trade-offs
        trade_offs = self._identify_trade_offs()
        
        # Combine into explanation data
        explanation_data = {
            "objective_weights": objective_weights,
            "results_summary": {
                "objective_value": self.last_results.get("objective_value", 0),
                "component_values": self.last_results.get("component_values", {}),
                "key_metrics": key_metrics
            },
            "key_decisions": key_decisions,
            "trade_offs": trade_offs
        }
        
        return explanation_data
    
    def _extract_key_decisions(self) -> List[Dict[str, Any]]:
        """Extract key decisions from optimization results"""
        # This would implement logic to identify the most impactful decisions
        # For now, return placeholder data
        key_decisions = []
        
        # Extract from AP decisions
        ap_decisions = self.last_results.get("ap_decisions", [])
        for i, decision in enumerate(ap_decisions[:5]):  # Just take first 5 for example
            key_decisions.append({
                "type": "accounts_payable",
                "id": f"ap_{i}",
                "description": decision.get("description", ""),
                "impact": decision.get("impact", 0)
            })
        
        # Extract from AR decisions
        ar_decisions = self.last_results.get("ar_decisions", [])
        for i, decision in enumerate(ar_decisions[:5]):  # Just take first 5 for example
            key_decisions.append({
                "type": "accounts_receivable",
                "id": f"ar_{i}",
                "description": decision.get("description", ""),
                "impact": decision.get("impact", 0)
            })
        
        return key_decisions
    
    def _identify_trade_offs(self) -> List[Dict[str, Any]]:
        """Identify trade-offs in the optimization results"""
        # In a real implementation, this would analyze decisions to find trade-offs
        # For now, return placeholder data
        trade_offs = [
            {
                "description": "Early payment discounts vs. cash retention",
                "option1": "Capture all early payment discounts",
                "option2": "Retain cash for longer period",
                "decision": "Partial discount capture",
                "rationale": "Balanced approach based on cash needs"
            },
            {
                "description": "Aggressive collection vs. customer relationships",
                "option1": "Accelerate collections for all customers",
                "option2": "Maintain standard terms for relationship preservation",
                "decision": "Segmented approach by customer tier",
                "rationale": "Prioritize relationships with strategic customers"
            }
        ]
        
        return trade_offs
    
    def _prepare_decision_explanation_data(self, decision_id: str) -> Dict[str, Any]:
        """Prepare data for explaining a specific decision"""
        # This would extract data about the specific decision to explain
        # For now, return placeholder data
        decision_data = {
            "decision_id": decision_id,
            "decision_type": "payment_timing" if decision_id.startswith("ap_") else "collection_action",
            "decision_details": {"placeholder": "actual decision details would go here"},
            "alternatives": [{"alternative": "option 1"}, {"alternative": "option 2"}],
            "constraints": {"cash_flow": "constraint details"}
        }
        
        return decision_data
    
    def _prepare_rca_explanation_data(self, outcome_id: str) -> Dict[str, Any]:
        """Prepare data for root cause analysis of a specific outcome"""
        # This would extract data about the specific outcome for RCA
        # For now, return placeholder data
        rca_data = {
            "outcome": "Cash shortfall in week 3",
            "context": {"cash_position": "details about cash position"},
            "related_factors": [{"factor": "Large supplier payment"}, {"factor": "Delayed customer payment"}],
            "historical_data": {"payment_patterns": "historical payment pattern data"}
        }
        
        return rca_data
    
    def run_scenario_analysis(self, data: Dict[str, Any], 
                           scenarios: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run scenario analysis with multiple parameter sets.
        
        Args:
            data: Base input data
            scenarios: List of scenario configurations
            
        Returns:
            Results for each scenario
        """
        scenario_results = []
        
        for i, scenario in enumerate(scenarios):
            # Create scenario name if not provided
            scenario_name = scenario.get("name", f"Scenario {i+1}")
            self.logger.info(f"Running scenario: {scenario_name}")
            
            # Update config with scenario parameters
            scenario_config = self.config.copy()
            for key, value in scenario.get("parameters", {}).items():
                if key in scenario_config["wc_optimization"]:
                    scenario_config["wc_optimization"][key] = value
                    
            # Create temporary optimizer with scenario config
            temp_optimizer = WorkingCapitalOptimizer(scenario_config["wc_optimization"])
            
            # Run optimization
            processed_data = temp_optimizer.preprocess_data(data)
            temp_optimizer.build_model(processed_data)
            temp_optimizer.solve()
            results = temp_optimizer.extract_solution()
            
            # Add scenario metadata
            results["scenario_name"] = scenario_name
            results["scenario_description"] = scenario.get("description", "")
            results["scenario_parameters"] = scenario.get("parameters", {})
            
            # Store results
            scenario_results.append(results)
        
        return {"scenarios": scenario_results}
    
    def compare_scenarios(self, scenario1_id: str, scenario2_id: str) -> Dict[str, Any]:
        """
        Generate AI explanation comparing two scenarios.
        
        Args:
            scenario1_id: ID of first scenario
            scenario2_id: ID of second scenario
            
        Returns:
            Comparative explanation
        """
        # This would implement scenario comparison logic using the AI explanation engine
        # For now, return placeholder data
        comparison = {
            "summary": "Comparison between scenarios",
            "key_differences": [
                "Difference 1",
                "Difference 2"
            ],
            "recommendation": "Recommended scenario based on analysis"
        }
        
        return comparison
