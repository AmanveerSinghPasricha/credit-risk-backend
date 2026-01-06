import logging
import pickle
import json
import pandas as pd
import shap
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from cors import setup_cors
from rate_limit import limiter
from schema import CreditRiskRequest, ApiResponse, CreditRiskPrediction

from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(
    title="Credit Risk Engine API",
    description="Production-grade credit risk prediction service",
    version="1.0.0"
)

setup_cors(app)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent.parent

# MODEL_PATH = BASE_DIR / "credit-risk-management" / "model" / "credit_risk_model.pkl"
# RF_PIPELINE_PATH = BASE_DIR / "credit-risk-management" / "model" / "rf_pipeline.pkl"
# SHAP_BACKGROUND_PATH = BASE_DIR / "credit-risk-management" / "model" / "shap_background.csv"

# MODEL_PATH = BASE_DIR  / "model" / "credit_risk_model.pkl"
# RF_PIPELINE_PATH = BASE_DIR  / "model" / "rf_pipeline.pkl"
# SHAP_BACKGROUND_PATH = BASE_DIR  / "model" / "shap_background.csv"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "credit_risk_model.pkl"
RF_PIPELINE_PATH = BASE_DIR / "model" / "rf_pipeline.pkl"
SHAP_BACKGROUND_PATH = BASE_DIR / "model" / "shap_background.csv"

BEST_THRESHOLD = 0.42

FEATURE_NAME_MAP = {
    "num__person_age": "Applicant age",
    "num__person_income": "Applicant income",
    "num__person_emp_length": "Employment length",
    "num__loan_amnt": "Loan amount",
    "num__loan_int_rate": "Interest rate",
    "num__loan_percent_income": "Loan burden relative to income",
    "num__cb_person_cred_hist_length": "Credit history length",

    "ord__loan_grade": "Loan grade",

    "bin__cb_person_default_on_file": "Previous default history",

    # One-hot encoded
    "ohe__person_home_ownership_RENT": "Living in rented accommodation",
    "ohe__person_home_ownership_OWN": "Owning a house",
    "ohe__person_home_ownership_MORTGAGE": "Home under mortgage",

    "ohe__loan_intent_EDUCATION": "Loan for education",
    "ohe__loan_intent_MEDICAL": "Loan for medical expenses",
    "ohe__loan_intent_PERSONAL": "Personal loan",
    "ohe__loan_intent_VENTURE": "Loan for business venture",
    "ohe__loan_intent_HOMEIMPROVEMENT": "Loan for home improvement",
    "ohe__loan_intent_DEBTCONSOLIDATION": "Loan for debt consolidation"
}

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    voting_model = pickle.load(f)
logger.info("VotingClassifier loaded")

with open(RF_PIPELINE_PATH, "rb") as f:
    rf_pipeline = pickle.load(f)

if not isinstance(rf_pipeline, ImbPipeline):
    raise RuntimeError("rf_pipeline.pkl must be an ImbPipeline")

logger.info("RF pipeline loaded")

# -------------------------------------------------
# EXTRACT PREPROCESSOR & RF MODEL
# -------------------------------------------------
preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_model = rf_pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

# -------------------------------------------------
# LOAD & TRANSFORM SHAP BACKGROUND (NUMERIC ONLY)
# -------------------------------------------------
shap_background_raw = pd.read_csv(SHAP_BACKGROUND_PATH)
shap_background = preprocessor.transform(shap_background_raw)

logger.info("SHAP background transformed")

# -------------------------------------------------
# INIT SHAP EXPLAINER (CORRECT)
# -------------------------------------------------
explainer = shap.TreeExplainer(
    rf_model,
    data=shap_background,
    feature_perturbation="interventional"
)

logger.info("SHAP explainer initialized")

# -------------------------------------------------
# SHAP → HUMAN READABLE
# -------------------------------------------------
def shap_to_english(top_factors: dict) -> list[str]:
    explanations = []

    for feature, value in top_factors.items():
        human_name = FEATURE_NAME_MAP.get(feature, feature)

        direction = (
            "increased the risk of default"
            if value > 0
            else "reduced the risk of default"
        )

        explanations.append(f"{human_name} {direction}.")

    return explanations

# -------------------------------------------------
# CACHED CORE INFERENCE
# -------------------------------------------------
@lru_cache(maxsize=1000)
def cached_predict(payload_json: str):
    payload = json.loads(payload_json)
    input_df = pd.DataFrame([payload])

    # ---- Prediction (VotingClassifier uses RAW input)
    probability = float(voting_model.predict_proba(input_df)[:, 1][0])
    decision = "REJECT" if probability >= BEST_THRESHOLD else "APPROVE"

    # ---- SHAP (TRANSFORM FIRST)
    input_transformed = preprocessor.transform(input_df)
    shap_values = explainer(
        input_transformed,
        check_additivity=False  
    )

    shap_class_1 = shap_values.values[0][:, 1]
    shap_dict = dict(zip(feature_names, shap_class_1))

    return probability, decision, shap_dict

# -------------------------------------------------
# RATE LIMITING
# -------------------------------------------------
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"error_code": "RATE_LIMIT", "message": "Too many prediction requests. Please wait and try again."})

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "Credit Risk Engine API running"}

# -------------------------------------------------
# PREDICTION ENDPOINT
# -------------------------------------------------
@app.post("/predict", response_model=ApiResponse)
@limiter.limit("10/minute")
def predict_credit_risk(request: Request, payload: CreditRiskRequest):
    try:
        data = payload.model_dump()

        # -------------------------------------------------
        # HARD BUSINESS VALIDATIONS
        # -------------------------------------------------
        age = data["person_age"]

        if data["person_emp_length"] > age - 18:
            raise HTTPException(
                status_code=400,
                detail="Employment length cannot exceed (age - 18)"
            )

        if data["cb_person_cred_hist_length"] > age - 18:
            raise HTTPException(
                status_code=400,
                detail="Credit history length cannot exceed (age - 18)"
            )

        if data["person_income"] <= 0:
            raise HTTPException(
                status_code=400,
                detail="Income must be greater than zero"
            )

        # -------------------------------------------------
        # DERIVED FIELD (AUTHORITATIVE)
        # -------------------------------------------------
        data["loan_percent_income"] = (
            data["loan_amnt"] / data["person_income"]
        )

        payload_json = json.dumps(data, sort_keys=True)

        probability, decision, shap_dict = cached_predict(payload_json)

        top_factors = dict(
            sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
        )

        explanation = shap_to_english(top_factors)

        prediction = CreditRiskPrediction(
            default_probability=round(probability, 4),
            decision=decision,
            explanation=explanation
        )

        return {
            "status": "success",
            "status_code": 200,
            "message": "Prediction generated successfully",
            "data": prediction.model_dump()
        }

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "status_code": 500,
                "message": "Prediction error occurred",
                "data": None
            }
        )
