"""
Prompt templates for the AI explanation engine
"""

# Prompt for general explanation of optimization results
GENERAL_EXPLANATION_PROMPT = """
Please explain the following working capital optimization results in business terms:

Objective Weights:
- Liquidity (α): {alpha}
- Financing Cost (β): {beta}
- Transaction Cost (γ): {gamma}
- Relationship Value (θ): {theta}

Key Results:
- Overall objective value: {objective_value}
- Component contributions: {component_values}
- Key metrics: {key_metrics}

Key Decisions:
{key_decisions}

Identified Trade-offs:
{trade_offs}

Provide a concise, business-focused explanation of these results, highlighting the most important insights and recommendations. Structure your response with:
1. A brief summary
2. Key insights (labeled as "KEY INSIGHT:")
3. Specific recommendations (labeled as "RECOMMENDATION:")
4. Important trade-offs (labeled as "TRADE-OFF:")
"""

# Prompt for explaining specific decisions
DECISION_EXPLANATION_PROMPT = """
Please explain the following specific decision from the working capital optimization:

Decision ID: {decision_id}
Decision Type: {decision_type}
Decision Details: {decision_details}

Alternative Options Considered: {alternatives}

Key Constraints: {constraints}

Explain why this specific decision was made, what factors influenced it the most, and what trade-offs were considered. Focus on the business implications and value.
"""

# Prompt for root cause analysis
RCA_PROMPT = """
Please analyze the root causes of the following outcome from our working capital optimization:

Outcome: {outcome}

Context: {context}

Related Factors: {related_factors}

Historical Data: {historical_data}

Perform a detailed root cause analysis explaining:
1. What were the primary and secondary factors contributing to this outcome?
2. How did these factors interact with each other?
3. Which factors could be controlled vs. which were external?
4. What early warning indicators could have predicted this outcome?
5. What actions could have been taken to change this outcome?
"""

# Prompt for scenario comparison
SCENARIO_COMPARISON_PROMPT = """
Please compare the following optimization scenarios:

Scenario 1: {scenario1_name}
- Objective value: {scenario1_objective}
- Key metrics: {scenario1_metrics}
- Key decisions: {scenario1_decisions}

Scenario 2: {scenario2_name}
- Objective value: {scenario2_objective}
- Key metrics: {scenario2_metrics}
- Key decisions: {scenario2_decisions}

Compare and contrast these scenarios, focusing on:
1. Major differences in outcomes
2. Key decision differences
3. Risk-reward trade-offs
4. Which scenario is preferable under what business conditions
"""

# Prompt for natural language strategy summary
STRATEGY_SUMMARY_PROMPT = """
Create a concise executive summary of the following working capital optimization strategy:

Optimization Objectives: {objectives}

Key AP Strategy Elements: {ap_strategy}

Key AR Strategy Elements: {ar_strategy}

Cash Flow Impact: {cash_flow_impact}

Relationship Impact: {relationship_impact}

Write a brief, executive-level summary that explains this strategy in business terms, highlighting the key benefits, risks, and implementation considerations. This should be suitable for a CFO or finance director to quickly understand the value proposition.
"""
