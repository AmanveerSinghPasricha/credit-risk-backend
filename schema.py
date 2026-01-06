from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, Dict, Any, List

# ======================================================
# REQUEST SCHEMA
# ======================================================
class CreditRiskRequest(BaseModel):
    """
    Input schema for credit risk prediction.
    Acts as the backend validation gate.
    """

    # ---------- Numeric features ----------

    person_age: int = Field(
        ..., ge=18, le=100, example=32
    )

    person_income: float = Field(
        ..., gt=0, example=85000
    )

    person_emp_length: float = Field(
        ..., ge=0, le=60, example=7
    )

    loan_amnt: float = Field(
        ..., gt=0, example=15000
    )

    loan_int_rate: float = Field(
        ..., gt=0, le=100, example=12.4
    )

    # Fraction of income (0–1)
    # loan_percent_income: float = Field(
    #     ..., ge=0, le=1, example=0.18
    # )

    loan_percent_income: Optional[float] = Field(
        None, ge=0, le=1, example=0.18
    )

    cb_person_cred_hist_length: float = Field(
        ..., ge=0, le=80, example=11
    )

    # ---------- Categorical features ----------

    cb_person_default_on_file: Literal[0, 1] = Field(
        ..., example=0
    )

    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        ..., example="B"
    )

    person_home_ownership: Literal[
        "RENT", "OWN", "MORTGAGE", "OTHER"
    ] = Field(
        ..., example="RENT"
    )

    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ] = Field(
        ..., example="EDUCATION"
    )

    # ---------- Field-level normalization ----------

    @field_validator(
        "loan_grade",
        "person_home_ownership",
        "loan_intent",
        mode="before"
    )
    @classmethod
    def normalize_strings(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Invalid categorical value")
        return value.upper()

    # ---------- Cross-field validation ----------

    @model_validator(mode="after")
    def validate_credit_history_vs_age(self):
        if self.cb_person_cred_hist_length > self.person_age:
            raise ValueError(
                "Credit history length cannot exceed applicant age"
            )
        return self

# ======================================================
# RESPONSE SCHEMAS
# ======================================================
class CreditRiskPrediction(BaseModel):
    default_probability: float
    decision: Literal["APPROVE", "REJECT"]
    explanation: List[str]

    
class ApiResponse(BaseModel):
    status: Literal["success", "error"]
    status_code: int
    message: str
    data: Optional[Dict[str, Any]] = None
