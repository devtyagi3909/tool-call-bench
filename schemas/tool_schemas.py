"""
Complex Pydantic v2 Schemas for Benchmarking LLM Tool-Calling Compliance.
Designed to stress-test frontier and open-weight models against:
1. Deeply nested recursive structures
2. Strict regex patterns and Enum constraints
3. Reasoning boundary token leakage detection
4. Schema violations and extra-field hallucinations
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"


class OrderItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(..., description="Unique alphanumeric SKU pattern (e.g. APX-1042)", pattern=r"^[A-Z]{3}-\d{4}$")
    quantity: int = Field(..., ge=1, le=100)
    unit_price: float = Field(..., gt=0.0)
    discount_pct: Optional[float] = Field(default=0.0, ge=0.0, le=50.0)


class ShippingAddress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    street: str = Field(..., min_length=3)
    city: str = Field(..., min_length=2)
    state_or_province: str = Field(..., min_length=2)
    postal_code: str = Field(..., min_length=4, max_length=10)
    country_iso: str = Field(..., min_length=2, max_length=2)


class CreateOrderPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., pattern=r"^CUST-\d{3,6}$")
    currency: Currency
    items: List[OrderItem] = Field(..., min_length=1)
    shipping_address: ShippingAddress
    special_instructions: Optional[str] = None
    require_signature: bool = False


class RefundDisputePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(..., pattern=r"^TX-\d{5,8}$")
    dispute_reason: str = Field(..., min_length=5)
    refund_amount: float = Field(..., gt=0.0)
    auto_approve: bool = False
    evidence_urls: List[str] = Field(default_factory=list)

    @field_validator("transaction_id")
    @classmethod
    def check_boundary_leakage(cls, v: str) -> str:
        boundary_markers = ["<think>", "</think>", "<|observation|>", "<|tool_call|>", "<|im_end|>"]
        for marker in boundary_markers:
            if marker in v:
                raise ValueError(f"Reasoning boundary token leakage detected in transaction_id: '{marker}'")
        return v


class FinancialReconciliationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement_id: str = Field(..., pattern=r"^STM-\d{4}-[A-Z]{3}$")
    period_start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    period_end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    total_credits: float = Field(..., ge=0.0)
    total_debits: float = Field(..., ge=0.0)
    variance_tolerance_pct: float = Field(default=0.01, ge=0.0, le=5.0)
    reconciled_by: str = Field(..., min_length=3)


class DatabaseMigrationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    migration_id: str = Field(..., pattern=r"^MIG-\d{6}$")
    target_database: str = Field(..., min_length=3)
    dry_run: bool = True
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
    rollback_script: Optional[str] = None
