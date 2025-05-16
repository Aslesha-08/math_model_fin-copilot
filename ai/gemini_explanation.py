"""
Gemini-powered explanation engine for working capital optimization results
"""
import os
import json
import time
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Import prompt templates
from .prompts import (
    GENERAL_EXPLANATION_PROMPT,
    DECISION_EXPLANATION_PROMPT,
    RCA_PROMPT
)

class GeminiExplanationEngine:
    """
    Explanation engine that uses Google's Gemini to generate human-readable
    explanations of optimization results.
    """
    def __init__(self, model="gemini-2.5-pro-preview-03-25"):
        """
        Initialize the Gemini-powered explanation engine.
        
        Args:
            model: The Gemini model to use
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variables
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
            
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Set up the model
        self.model = model
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Initialize Gemini model
        self.gemini = genai.GenerativeModel(
            model_name=self.model,
            generation_config=self.generation_config
        )
    
    def _extract_component_values(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract component values from optimization results.
        
        Args:
            results: Optimization results
            
        Returns:
            Component values
        """
        component_values = {
            "liquidity_value": 0,
            "financing_cost": 0,
            "transaction_cost": 0,
            "relationship_value": 0
        }
        
        # Extract metrics
        key_metrics = results.get("key_metrics", {})
        
        # Estimate component values
        if "cash_conversion_cycle" in key_metrics:
            # Working capital optimization
            component_values["liquidity_value"] = key_metrics.get("total_receivables", 0) - key_metrics.get("total_payables", 0)
            component_values["financing_cost"] = key_metrics.get("financing_costs", 0)
            component_values["transaction_cost"] = key_metrics.get("ap_penalties_paid", 0) + key_metrics.get("ar_action_costs", 0)
            
            # Relationship value is harder to quantify, use a placeholder
            component_values["relationship_value"] = 5000
        elif "total_paid" in key_metrics:
            # AP optimization
            component_values["liquidity_value"] = key_metrics.get("average_days_paid", 0) * 1000  # Rough estimate
            component_values["financing_cost"] = key_metrics.get("financing_costs", 0)
            component_values["transaction_cost"] = key_metrics.get("total_penalties_paid", 0)
            component_values["relationship_value"] = key_metrics.get("total_discounts_captured", 0) * 2  # Estimate
        elif "total_expected_collections" in key_metrics:
            # AR optimization
            component_values["liquidity_value"] = key_metrics.get("total_expected_collections", 0)
            component_values["financing_cost"] = 0  # No direct financing in AR
            component_values["transaction_cost"] = key_metrics.get("total_action_costs", 0)
            component_values["relationship_value"] = 5000 - (key_metrics.get("average_collection_time", 0) * 10)  # Estimate
            
        return component_values
    
    def _extract_key_decisions(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract key decisions from optimization results.
        
        Args:
            results: Optimization results
            
        Returns:
            List of key decisions
        """
        key_decisions = []
        
        # Extract AP decisions
        if "ap_decisions" in results:
            ap_decisions = results["ap_decisions"]
            
            # Count decision types
            early_with_discount = 0
            early_no_discount = 0
            on_time = 0
            late = 0
            
            for decision in ap_decisions:
                payment_timing = decision.get("payment_timing", "")
                if payment_timing == "early_with_discount":
                    early_with_discount += 1
                elif payment_timing == "early_no_discount":
                    early_no_discount += 1
                elif payment_timing == "on_time":
                    on_time += 1
                elif payment_timing == "late":
                    late += 1
            
            total_ap = len(ap_decisions)
            if total_ap > 0:
                key_decisions.append(f"Pay {early_with_discount} invoices ({early_with_discount/total_ap:.1%}) early to capture discounts")
                key_decisions.append(f"Pay {early_no_discount} invoices ({early_no_discount/total_ap:.1%}) early without discounts")
                key_decisions.append(f"Pay {on_time} invoices ({on_time/total_ap:.1%}) on time")
                key_decisions.append(f"Pay {late} invoices ({late/total_ap:.1%}) late")
                
        # Extract AR decisions
        if "ar_decisions" in results:
            ar_decisions = results["ar_decisions"]
            
            # Count action types
            action_counts = {}
            before_due = 0
            after_due = 0
            
            for decision in ar_decisions:
                # Count timing
                timing = decision.get("collection_timing", "")
                if timing == "before_due":
                    before_due += 1
                elif timing == "after_due":
                    after_due += 1
                
                # Count actions
                for action in decision.get("actions", []):
                    action_counts[action] = action_counts.get(action, 0) + 1
            
            # Add timing decisions
            total_ar = len(ar_decisions)
            if total_ar > 0:
                key_decisions.append(f"Expect to collect {before_due} invoices ({before_due/total_ar:.1%}) before due date")
                key_decisions.append(f"Expect to collect {after_due} invoices ({after_due/total_ar:.1%}) after due date")
            
            # Add action decisions
            for action, count in action_counts.items():
                key_decisions.append(f"Use '{action}' action on {count} invoices ({count/total_ar:.1%})")
                
        return key_decisions
    
    def _extract_trade_offs(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract trade-offs from optimization results.
        
        Args:
            results: Optimization results
            
        Returns:
            List of trade-offs
        """
        trade_offs = []
        
        # Add common trade-offs
        trade_offs.append("Cash flow optimization vs. supplier/customer relationship management")
        trade_offs.append("Short-term cost savings vs. long-term strategic benefits")
        trade_offs.append("Operational efficiency vs. finance department workload")
        
        # Check if we have AP optimization results
        if "ap_decisions" in results:
            trade_offs.append("Early payment discounts vs. holding cash longer")
            trade_offs.append("Payment timing vs. supplier relationship quality")
        
        # Check if we have AR optimization results
        if "ar_decisions" in results:
            trade_offs.append("Collection effort costs vs. faster cash inflow")
            trade_offs.append("Aggressive collection tactics vs. customer satisfaction")
        
        # Check if we have integrated working capital results
        if "key_metrics" in results and "cash_conversion_cycle" in results["key_metrics"]:
            trade_offs.append("Accounts payable extension vs. accounts receivable acceleration")
        
        return trade_offs
    
    def _prepare_prompt_variables(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare variables for prompt templates.
        
        Args:
            results: Optimization results
            
        Returns:
            Prompt variables
        """
        # Extract objective weights
        objective_weights = {
            "alpha": 1.0,  # Liquidity
            "beta": 0.8,   # Financing Cost
            "gamma": 0.6,  # Transaction Cost
            "theta": 0.7   # Relationship Value
        }
        
        # Try to extract from results if available
        if "optimization_mode" in results:
            mode = results["optimization_mode"]
            if mode == "cash_flow" or mode == "cash":
                objective_weights = {"alpha": 1.0, "beta": 0.8, "gamma": 0.6, "theta": 0.4}
            elif mode == "relationship" or mode == "supplier" or mode == "customer":
                objective_weights = {"alpha": 0.7, "beta": 0.6, "gamma": 0.5, "theta": 1.0}
            elif mode == "cost":
                objective_weights = {"alpha": 0.7, "beta": 1.0, "gamma": 1.0, "theta": 0.5}
            elif mode == "balanced":
                objective_weights = {"alpha": 0.8, "beta": 0.8, "gamma": 0.8, "theta": 0.8}
        
        # Prepare variables
        variables = {
            "alpha": objective_weights["alpha"],
            "beta": objective_weights["beta"],
            "gamma": objective_weights["gamma"],
            "theta": objective_weights["theta"],
            "objective_value": results.get("objective_value", 0),
            "component_values": self._extract_component_values(results),
            "key_metrics": results.get("key_metrics", {}),
            "key_decisions": self._extract_key_decisions(results),
            "trade_offs": self._extract_trade_offs(results)
        }
        
        return variables
    
    def _format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Format prompt template with variables.
        
        Args:
            template: Prompt template
            variables: Variables to insert into template
            
        Returns:
            Formatted prompt
        """
        # Convert lists to bullet points
        for key, value in variables.items():
            if isinstance(value, list):
                variables[key] = "\n".join([f"- {item}" for item in value])
            elif isinstance(value, dict):
                variables[key] = "\n".join([f"- {k}: {v}" for k, v in value.items()])
        
        # Format the prompt
        return template.format(**variables)
    
    def _generate_fallback_explanation(self, results: Dict[str, Any]) -> str:
        """
        Generate a fallback explanation when Gemini API fails.
        
        Args:
            results: Optimization results
            
        Returns:
            Fallback explanation text
        """
        objective_value = results.get("objective_value", "N/A")
        key_metrics_str = json.dumps(results.get("key_metrics", {}), indent=2)
        
        fallback = f"""
Here's a summary of the optimization results instead:

Objective Value: {objective_value}
Key Metrics: {key_metrics_str}

Summary: The optimization found a solution that balances cost, timing, and relationship factors.
"""
        return fallback
    
    def _extract_text_from_response(self, response) -> str:
        """
        Safely extract text from Gemini API response, handling different response types.
        
        Args:
            response: Gemini API response object
            
        Returns:
            Extracted text from response
        """
        try:
            # Direct text property (most common case)
            if hasattr(response, 'text'):
                return response.text
            
            # List of parts
            if hasattr(response, 'parts') and isinstance(response.parts, list):
                parts_text = []
                for part in response.parts:
                    if hasattr(part, 'text'):
                        parts_text.append(part.text)
                if parts_text:
                    return "\n".join(parts_text)
            
            # Single part object
            if hasattr(response, 'parts') and hasattr(response.parts, 'text'):
                return response.parts.text
            
            # Response with candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate_texts = []
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                candidate_texts.append(part.text)
                if candidate_texts:
                    return "\n".join(candidate_texts)
            
            # Direct content access
            if hasattr(response, 'content'):
                if isinstance(response.content, str):
                    return response.content
                if hasattr(response.content, 'parts'):
                    parts_texts = []
                    for part in response.content.parts:
                        if hasattr(part, 'text'):
                            parts_texts.append(part.text)
                    if parts_texts:
                        return "\n".join(parts_texts)
            
            # Last resort: convert to string
            return str(response)
            
        except Exception as e:
            return f"Error extracting text from response: {str(e)}"
    
    def generate_explanation(self, 
                           results: Dict[str, Any], 
                           explanation_type: str = "general",
                           specific_entity: Optional[str] = None,
                           specific_question: Optional[str] = None) -> str:
        """
        Generate explanation for optimization results.
        
        Args:
            results: Optimization results
            explanation_type: Type of explanation to generate
            specific_entity: Specific entity to explain (e.g., invoice ID)
            specific_question: Specific question to answer
            
        Returns:
            Explanation text
        """
        try:
            # Prepare variables for prompt
            prompt_variables = self._prepare_prompt_variables(results)
            
            # Add specific entity or question if provided
            if specific_entity:
                prompt_variables["specific_entity"] = specific_entity
            if specific_question:
                prompt_variables["specific_question"] = specific_question
            
            # Select the appropriate prompt template
            if explanation_type == "general":
                template = GENERAL_EXPLANATION_PROMPT
            elif explanation_type == "decision":
                template = DECISION_EXPLANATION_PROMPT
                
                # Find the specific decision
                if specific_entity:
                    if "ap_decisions" in results:
                        for decision in results["ap_decisions"]:
                            if decision.get("invoice_id") == specific_entity:
                                prompt_variables["decision_id"] = specific_entity
                                prompt_variables["decision_type"] = "Accounts Payable"
                                prompt_variables["decision_details"] = decision
                                prompt_variables["alternatives"] = "Pay earlier, pay on time, pay later"
                                prompt_variables["constraints"] = "Cash flow, supplier relationship, discount opportunities"
                                break
                    
                    if "ar_decisions" in results and "decision_id" not in prompt_variables:
                        for decision in results["ar_decisions"]:
                            if decision.get("invoice_id") == specific_entity:
                                prompt_variables["decision_id"] = specific_entity
                                prompt_variables["decision_type"] = "Accounts Receivable"
                                prompt_variables["decision_details"] = decision
                                prompt_variables["alternatives"] = "Different collection actions or timing"
                                prompt_variables["constraints"] = "Customer relationship, collection costs, probability of payment"
                                break
                
            elif explanation_type == "root_cause":
                template = RCA_PROMPT
            else:
                raise ValueError(f"Unknown explanation type: {explanation_type}")
            
            # Format the prompt
            prompt = self._format_prompt(template, prompt_variables)
            
            # Generate explanation with retry logic
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Generate content
                    response = self.gemini.generate_content(prompt)
                    
                    # Extract text using our safe extraction method
                    explanation_text = self._extract_text_from_response(response)
                    
                    # Check if we got a valid response
                    if explanation_text and len(explanation_text) > 20:
                        return explanation_text
                    
                    # If response is too short, retry
                    time.sleep(1)  # Add delay between retries
                    
                except Exception as inner_e:
                    # Log the specific error
                    print(f"Retry {retry+1}/{max_retries}: Error in Gemini API: {str(inner_e)}")
                    time.sleep(2)  # Longer delay after an error
            
            # If all retries failed, return a fallback explanation
            return self._generate_fallback_explanation(results)
            
        except Exception as e:
            # If an error occurs, return a fallback explanation
            error_message = f"Error generating explanation: {str(e)}\n\n"
            fallback = error_message + self._generate_fallback_explanation(results)
            return fallback
