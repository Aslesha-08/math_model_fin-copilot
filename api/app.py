"""
FastAPI web interface for the Working Capital Optimization system
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.ap_optimizer import APOptimizer
from optimization.ar_optimizer import AROptimizer
from optimization.wc_optimizer import WorkingCapitalOptimizer
from ai.explanation import ExplanationEngine

# Initialize the FastAPI app
app = FastAPI(
    title="Working Capital Optimization API",
    description="API for optimizing accounts payable, accounts receivable, and working capital",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response validation
class Invoice(BaseModel):
    id: str
    amount: float
    due_date: int
    supplier_id: Optional[str] = None
    customer_id: Optional[str] = None
    supplier_priority: Optional[str] = None
    customer_segment: Optional[str] = None
    discount_date: Optional[int] = None
    discount_rate: Optional[float] = None
    penalty_rate: Optional[float] = None
    issue_date: Optional[int] = None
    payment_probability: Optional[float] = None

class CashPosition(BaseModel):
    initial_balance: float
    min_balance: Optional[float] = None

class Forecast(BaseModel):
    inflow: float = 0
    outflow: float = 0

class APOptimizationRequest(BaseModel):
    ap_invoices: List[Invoice]
    cash_position: CashPosition
    forecasts: Dict[int, Forecast] = {}
    optimization_mode: str = "cost"  # cost, supplier, cash, custom
    custom_weights: Optional[Dict[str, float]] = None
    horizon: int = 90
    max_borrowing: Optional[float] = None
    borrowing_rate: float = 0.0001  # Daily rate

class AROptimizationRequest(BaseModel):
    ar_invoices: List[Invoice]
    cash_position: CashPosition
    forecasts: Dict[int, Forecast] = {}
    optimization_mode: str = "cash_flow"  # cash_flow, customer, financing, custom
    custom_weights: Optional[Dict[str, float]] = None
    collection_actions: Optional[List[str]] = None
    horizon: int = 90
    max_financing: Optional[float] = None
    financing_rate: float = 0.0001  # Daily rate

class WCOptimizationRequest(BaseModel):
    ap_invoices: List[Invoice]
    ar_invoices: List[Invoice]
    cash_position: CashPosition
    forecasts: Dict[int, Forecast] = {}
    optimization_mode: str = "balanced"  # cash_flow, balanced, relationship, cost, custom
    custom_weights: Optional[Dict[str, float]] = None
    horizon: int = 90
    max_financing: Optional[float] = None
    financing_rate: float = 0.0001  # Daily rate
    ap_weights: Optional[Dict[str, float]] = None
    ar_weights: Optional[Dict[str, float]] = None

class ExplanationRequest(BaseModel):
    optimization_results: Dict[str, Any]
    explanation_type: str = "general"  # general, decision, root_cause
    specific_entity: Optional[str] = None  # invoice_id, supplier_id, or customer_id
    specific_question: Optional[str] = None

# API Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Working Capital Optimization API"}

@app.post("/api/v1/optimize/ap")
async def optimize_ap(request: APOptimizationRequest):
    """
    Optimize accounts payable by determining optimal payment timing
    """
    try:
        # Prepare configuration
        config = {
            "optimization_mode": request.optimization_mode,
            "horizon": request.horizon,
            "max_borrowing": request.max_borrowing,
            "borrowing_rate": request.borrowing_rate
        }
        
        # Add custom weights if provided
        if request.custom_weights:
            config.update(request.custom_weights)
        
        # Initialize optimizer
        optimizer = APOptimizer(config)
        
        # Prepare data
        data = {
            "ap_invoices": [invoice.dict() for invoice in request.ap_invoices],
            "cash_position": request.cash_position.dict(),
            "forecasts": {int(k): v.dict() for k, v in request.forecasts.items()}
        }
        
        # Run optimization
        processed_data = optimizer.preprocess_data(data)
        optimizer.build_model(processed_data)
        results = optimizer.solve()
        
        # Add metadata
        results["metadata"] = {
            "optimization_timestamp": datetime.now().isoformat(),
            "optimization_mode": request.optimization_mode,
            "horizon": request.horizon
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/api/v1/optimize/ar")
async def optimize_ar(request: AROptimizationRequest):
    """
    Optimize accounts receivable by determining optimal collection strategies
    """
    try:
        # Prepare configuration
        config = {
            "optimization_mode": request.optimization_mode,
            "horizon": request.horizon,
            "max_financing": request.max_financing,
            "financing_rate": request.financing_rate
        }
        
        # Add collection actions if provided
        if request.collection_actions:
            config["collection_actions"] = request.collection_actions
        
        # Add custom weights if provided
        if request.custom_weights:
            config.update(request.custom_weights)
        
        # Initialize optimizer
        optimizer = AROptimizer(config)
        
        # Prepare data
        data = {
            "ar_invoices": [invoice.dict() for invoice in request.ar_invoices],
            "cash_position": request.cash_position.dict(),
            "forecasts": {int(k): v.dict() for k, v in request.forecasts.items()}
        }
        
        # Run optimization
        processed_data = optimizer.preprocess_data(data)
        optimizer.build_model(processed_data)
        results = optimizer.solve()
        
        # Add metadata
        results["metadata"] = {
            "optimization_timestamp": datetime.now().isoformat(),
            "optimization_mode": request.optimization_mode,
            "horizon": request.horizon
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/api/v1/optimize/wc")
async def optimize_working_capital(request: WCOptimizationRequest):
    """
    Optimize integrated working capital by coordinating AP and AR strategies
    """
    try:
        # Prepare configuration
        config = {
            "optimization_mode": request.optimization_mode,
            "horizon": request.horizon,
            "max_financing": request.max_financing,
            "financing_rate": request.financing_rate
        }
        
        # Add custom weights if provided
        if request.custom_weights:
            config.update(request.custom_weights)
        
        # Add specific weights for AP and AR if provided
        if request.ap_weights:
            config["ap_weights"] = request.ap_weights
        if request.ar_weights:
            config["ar_weights"] = request.ar_weights
        
        # Initialize optimizer
        optimizer = WorkingCapitalOptimizer(config)
        
        # Prepare data
        data = {
            "ap_invoices": [invoice.dict() for invoice in request.ap_invoices],
            "ar_invoices": [invoice.dict() for invoice in request.ar_invoices],
            "cash_position": request.cash_position.dict(),
            "forecasts": {int(k): v.dict() for k, v in request.forecasts.items()}
        }
        
        # Run optimization
        processed_data = optimizer.preprocess_data(data)
        optimizer.build_model(processed_data)
        results = optimizer.solve()
        
        # Add metadata
        results["metadata"] = {
            "optimization_timestamp": datetime.now().isoformat(),
            "optimization_mode": request.optimization_mode,
            "horizon": request.horizon
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/api/v1/explain")
async def generate_explanation(request: ExplanationRequest):
    """
    Generate human-readable explanations for optimization results
    """
    try:
        # Initialize explanation engine
        engine = ExplanationEngine()
        
        # Generate explanation
        explanation = engine.generate_explanation(
            request.optimization_results,
            explanation_type=request.explanation_type,
            specific_entity=request.specific_entity,
            specific_question=request.specific_question
        )
        
        return {
            "explanation": explanation,
            "metadata": {
                "explanation_timestamp": datetime.now().isoformat(),
                "explanation_type": request.explanation_type
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
