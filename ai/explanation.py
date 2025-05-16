"""
Explanation Engine for Working Capital Optimization using Generative AI
"""
import os
import json
from typing import Dict, List, Any, Optional
import openai
from dotenv import load_dotenv
from .prompts import GENERAL_EXPLANATION_PROMPT, DECISION_EXPLANATION_PROMPT, RCA_PROMPT

# Load environment variables
load_dotenv()

# Set API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")


class ExplanationEngine:
    """
    AI-powered explanation engine that translates optimization results into
    human-friendly explanations and provides root cause analysis.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the explanation engine.
        
        Args:
            config: Configuration for the explanation engine
        """
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 1000)
        self.last_results = None
        self.explanation_cache = {}
    
    def generate_explanation(self, optimization_data: Dict[str, Any], 
                            explanation_type: str = "general") -> Dict[str, Any]:
        """
        Generate an explanation based on optimization results.
        
        Args:
            optimization_data: Data from optimization results
            explanation_type: Type of explanation to generate
            
        Returns:
            Dictionary containing the structured explanation
        """
        # Store results for reference
        self.last_results = optimization_data
        
        # Check cache for existing explanation
        cache_key = f"{explanation_type}_{hash(json.dumps(optimization_data, sort_keys=True))}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # Select prompt based on explanation type
        if explanation_type == "general":
            prompt = self._create_general_explanation_prompt(optimization_data)
        elif explanation_type == "decision":
            prompt = self._create_decision_explanation_prompt(optimization_data)
        elif explanation_type == "rca":
            prompt = self._create_rca_explanation_prompt(optimization_data)
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")
        
        # Call LLM API
        response = self._call_llm_api(prompt)
        
        # Post-process the response
        structured_explanation = self._post_process_explanation(response, explanation_type)
        
        # Cache the explanation
        self.explanation_cache[cache_key] = structured_explanation
        
        return structured_explanation
    
    def _create_general_explanation_prompt(self, optimization_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create prompt for general explanation of optimization results.
        
        Args:
            optimization_data: Structured optimization results
            
        Returns:
            List of message dictionaries for the LLM prompt
        """
        # Extract key components from optimization data
        objective_weights = optimization_data.get("objective_weights", {})
        results_summary = optimization_data.get("results_summary", {})
        key_decisions = optimization_data.get("key_decisions", {})
        trade_offs = optimization_data.get("trade_offs", [])
        
        # Format the prompt with data
        formatted_prompt = GENERAL_EXPLANATION_PROMPT.format(
            alpha=objective_weights.get("liquidity", 0),
            beta=objective_weights.get("financing", 0),
            gamma=objective_weights.get("transaction", 0),
            theta=objective_weights.get("relationship", 0),
            objective_value=results_summary.get("objective_value", 0),
            component_values=json.dumps(results_summary.get("component_values", {})),
            key_metrics=json.dumps(results_summary.get("key_metrics", {})),
            key_decisions=json.dumps(key_decisions, indent=2),
            trade_offs=json.dumps(trade_offs, indent=2)
        )
        
        # Create message structure for API call
        prompt = [
            {"role": "system", "content": """You are an AI financial analyst specializing in working capital optimization. 
            Your job is to explain optimization results in clear, business-focused language. 
            Focus on key insights, trade-offs made, and the business impact of the recommended actions.
            Always explain WHY certain decisions were made based on the objective weights and constraints."""},
            {"role": "user", "content": formatted_prompt}
        ]
        
        return prompt
    
    def _create_decision_explanation_prompt(self, optimization_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create prompt for explaining specific decisions in the optimization.
        
        Args:
            optimization_data: Structured optimization results with decision data
            
        Returns:
            List of message dictionaries for the LLM prompt
        """
        # Extract decision-specific information
        decision_id = optimization_data.get("decision_id", "")
        decision_type = optimization_data.get("decision_type", "")
        decision_details = optimization_data.get("decision_details", {})
        alternatives = optimization_data.get("alternatives", [])
        constraints = optimization_data.get("constraints", {})
        
        # Format the prompt with decision data
        formatted_prompt = DECISION_EXPLANATION_PROMPT.format(
            decision_id=decision_id,
            decision_type=decision_type,
            decision_details=json.dumps(decision_details, indent=2),
            alternatives=json.dumps(alternatives, indent=2),
            constraints=json.dumps(constraints, indent=2)
        )
        
        # Create message structure for API call
        prompt = [
            {"role": "system", "content": """You are an AI financial analyst specializing in working capital optimization. 
            Your job is to explain specific optimization decisions in clear, business-focused language.
            Focus on why this particular choice was made over alternatives, and what trade-offs were considered."""},
            {"role": "user", "content": formatted_prompt}
        ]
        
        return prompt
    
    def _create_rca_explanation_prompt(self, optimization_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create prompt for root cause analysis of specific outcomes.
        
        Args:
            optimization_data: Structured optimization results with outcome data
            
        Returns:
            List of message dictionaries for the LLM prompt
        """
        # Extract RCA-specific information
        outcome = optimization_data.get("outcome", "")
        context = optimization_data.get("context", {})
        related_factors = optimization_data.get("related_factors", [])
        historical_data = optimization_data.get("historical_data", {})
        
        # Format the prompt with RCA data
        formatted_prompt = RCA_PROMPT.format(
            outcome=outcome,
            context=json.dumps(context, indent=2),
            related_factors=json.dumps(related_factors, indent=2),
            historical_data=json.dumps(historical_data, indent=2)
        )
        
        # Create message structure for API call
        prompt = [
            {"role": "system", "content": """You are an AI financial analyst specializing in working capital optimization. 
            Your job is to perform root cause analysis on specific outcomes from the optimization model.
            Identify the key factors that contributed to this outcome, their relative importance, and how they interacted."""},
            {"role": "user", "content": formatted_prompt}
        ]
        
        return prompt
    
    def _call_llm_api(self, prompt: List[Dict[str, str]]) -> str:
        """
        Call the LLM API with the constructed prompt.
        
        Args:
            prompt: List of message dictionaries for the LLM
            
        Returns:
            Generated response from the LLM
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def _post_process_explanation(self, response: str, explanation_type: str) -> Dict[str, Any]:
        """
        Post-process the LLM response into structured format.
        
        Args:
            response: Raw LLM response text
            explanation_type: Type of explanation
            
        Returns:
            Structured explanation
        """
        # Process based on explanation type
        if explanation_type == "general":
            return self._structure_general_explanation(response)
        elif explanation_type == "decision":
            return self._structure_decision_explanation(response)
        elif explanation_type == "rca":
            return self._structure_rca_explanation(response)
        else:
            return {"raw_explanation": response}
    
    def _structure_general_explanation(self, response: str) -> Dict[str, Any]:
        """
        Structure a general explanation into components.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured explanation with sections
        """
        sections = {
            "summary": "",
            "key_insights": [],
            "recommendations": [],
            "trade_offs": []
        }
        
        # Simple parsing logic
        current_section = "summary"
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "KEY INSIGHT" in line.upper() or "INSIGHT" in line.upper():
                current_section = "key_insights"
                continue
            elif "RECOMMENDATION" in line.upper():
                current_section = "recommendations"
                continue
            elif "TRADE-OFF" in line.upper() or "TRADEOFF" in line.upper():
                current_section = "trade_offs"
                continue
                
            if current_section == "summary":
                sections["summary"] += line + "\n"
            elif current_section in ["key_insights", "recommendations", "trade_offs"]:
                # Remove numbering if present
                cleaned_line = line
                if line[0].isdigit() and line[1:3] in ['. ', ') ', '- ']:
                    cleaned_line = line[3:].strip()
                elif line.startswith('- '):
                    cleaned_line = line[2:].strip()
                    
                if cleaned_line:
                    sections[current_section].append(cleaned_line)
        
        # Clean up summary (remove trailing newlines)
        sections["summary"] = sections["summary"].strip()
        
        return sections
    
    def _structure_decision_explanation(self, response: str) -> Dict[str, Any]:
        """
        Structure a decision explanation into components.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured explanation focused on a specific decision
        """
        # TODO: Implement more sophisticated parsing if needed
        explanation = {
            "decision_rationale": response,
            "factors": []  # Would extract specific factors in a more robust implementation
        }
        
        return explanation
    
    def _structure_rca_explanation(self, response: str) -> Dict[str, Any]:
        """
        Structure a root cause analysis explanation.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured RCA with root causes and contributing factors
        """
        # TODO: Implement more sophisticated parsing if needed
        explanation = {
            "root_cause_analysis": response,
            "primary_factors": [],  # Would extract specific factors in a more robust implementation
            "secondary_factors": []
        }
        
        return explanation
