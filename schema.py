"""Pydantic v2 models for Level 1 output (guide §5.1) and internal facts for Level 2.

Validators check format only. They never round, recalculate or correct values.
"""
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import provider_for

Number = int | float

ChargeType = Literal[
    "energy", "fixed", "fpa", "quarterly_adjustment", "surcharge", "meter_rent", "subsidy", "other"
]
TaxType = Literal["gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax"]
Provider = Literal["IESCO", "LESCO", "KE", "MEPCO", "FESCO", "GEPCO", "PESCO", "HESCO", "SEPCO",
                   "QESCO", "TESCO", "OTHER"]
PROVIDERS = set(Provider.__args__)
PROVIDER_ALIASES = {"KESC": "KE", "K-ELECTRIC": "KE", "K ELECTRIC": "KE", "KELECTRIC": "KE"}


class Charge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ChargeType
    amount: Number


class Tax(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: TaxType
    amount: Number


class Level1Bill(BaseModel):
    """Exactly the §5.1 fields, in §5.1 order."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider | None = None
    tariff: str | None = None
    sanctioned_load_kw: Number | None = None

    bill_month: str | None = None
    reading_date: str | None = None
    issue_date: str | None = None
    due_date: str | None = None

    previous_reading: Number | None = None
    current_reading: Number | None = None
    units_consumed: Number | None = None

    charges: list[Charge] = Field(default_factory=list)
    total_charges: Number | None = None

    taxes: list[Tax] = Field(default_factory=list)
    total_taxes: Number | None = None

    current_bill: Number | None = None
    arrears: Number | None = None
    payable_within_due_date: Number | None = None
    payable_after_due_date: Number | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _provider(cls, v):
        """Known DISCOs pass; any other non-empty value becomes OTHER (never a validation failure)."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        key = str(v).strip().upper()
        key = PROVIDER_ALIASES.get(key, key)
        return key if key in PROVIDERS else "OTHER"

    @field_validator("bill_month")
    @classmethod
    def _month_format(cls, v: str | None) -> str | None:
        if v is not None:
            datetime.strptime(v, "%Y-%m")  # raises ValueError if not YYYY-MM
            if len(v) != 7:
                raise ValueError("bill_month must be YYYY-MM")
        return v

    @field_validator("reading_date", "issue_date", "due_date")
    @classmethod
    def _date_format(cls, v: str | None) -> str | None:
        if v is not None:
            datetime.strptime(v, "%Y-%m-%d")
            if len(v) != 10:
                raise ValueError("date must be YYYY-MM-DD")
        return v


# ---- Internal facts for Level 2 (never exported) ----

class UsageMonth(BaseModel):
    month: str | None = None  # as printed or YYYY-MM
    units: Number | None = None
    bill_amount: Number | None = None
    payment: Number | None = None


class RateSlab(BaseModel):
    description: str | None = None  # e.g. "Variable Upto 100 Units"
    units: Number | None = None
    rate: Number | None = None
    amount: Number | None = None


class SplitItem(BaseModel):
    label: str | None = None  # e.g. "energy charges", "taxes"
    percent: Number | None = None


class LatePayment(BaseModel):
    label: str | None = None  # e.g. "after 24 Apr"
    amount: Number | None = None


class InternalFacts(BaseModel):
    usage_history: list[UsageMonth] = Field(default_factory=list)
    rates: list[RateSlab] = Field(default_factory=list)
    split: list[SplitItem] = Field(default_factory=list)
    late_payment_amounts: list[LatePayment] = Field(default_factory=list)
    validation_flags: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    level1: Level1Bill
    internal: InternalFacts = Field(default_factory=InternalFacts)


def to_level1_json(bill: Level1Bill) -> str:
    """One-line JSON with only the §5.1 keys, in §5.1 order, nulls included."""
    return json.dumps(bill.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def empty_level1(bill_id: str, provider: str | None = None) -> Level1Bill:
    """Fully keyed fallback: null scalars, [] lists. Provider comes from bill_id if not given."""
    return Level1Bill(provider=provider or provider_for(bill_id))
