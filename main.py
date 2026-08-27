# =======================================================================
# main.py - Student Academic Risk Intelligence System API
# =======================================================================

# --- Standard library / third-party imports ---
import pandas as pd
import numpy as np
import importlib
import uvicorn

# Import FastAPI dynamically so static analyzers that do not have the
# optional FastAPI package in their configured environment do not flag this
# module-level import as an unresolved import.
FastAPI = importlib.import_module("fastapi").FastAPI
pydantic = importlib.import_module("pydantic")
BaseModel = pydantic.BaseModel
Field = pydantic.Field
field_validator = pydantic.field_validator


# -----------------------------------------------------------------------
# Create the FastAPI application instance with metadata
# -----------------------------------------------------------------------
app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0",
)


# -----------------------------------------------------------------------
# load_data: loads the CSV and applies the same feature engineering
# used in analysis.py, so the API works with a consistently prepared
# DataFrame.
# -----------------------------------------------------------------------
def load_data():
    """
    Loads Maths.csv from the data/ folder and engineers the same
    additional features used in analysis.py.

    Returns
    -------
    pd.DataFrame
        The original data plus engineered feature columns.
    """

    # Load the raw CSV file
    df = pd.read_csv("data/Maths.csv")

    # --- Result: categorical outcome based on G3 ---
    # G3 == 0  -> "Dropout" (special case, NOT a low score)
    # G3 1-9   -> "Fail"
    # G3 10-20 -> "Pass"
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:
            return "Pass"

    df["Result"] = df["G3"].apply(classify_result)

    # --- Percentage: final grade expressed as a percentage of 20 ---
    df["Percentage"] = (df["G3"] / 20) * 100

    # --- avg_alcohol: average of workday and weekend alcohol consumption ---
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # --- parent_edu_avg: average education level of mother and father ---
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # --- grade_trend: change in performance from G1 to G3 ---
    df["grade_trend"] = df["G3"] - df["G1"]

    # --- total_support: count of "yes" responses across support columns ---
    support_cols = ["schoolsup", "famsup", "paid"]
    df["total_support"] = (df[support_cols] == "yes").sum(axis=1)

    # --- risk_score: composite score combining multiple risk factors ---
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # --- g1_g2_avg: average of first and second period grades ---
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    return df


# -----------------------------------------------------------------------
# Load and prepare the data once at startup, store it in a module-level
# variable so API endpoints can access it without reloading each time.
# -----------------------------------------------------------------------
df = load_data()

# -----------------------------------------------------------------------
# GET /summary
# Returns high-level summary statistics for the dataset.
# -----------------------------------------------------------------------
@app.get("/summary")
def get_summary():
    """
    Computes and returns overall class statistics:
    total students, class average G3, pass rate (non-dropouts only),
    at-risk count, and dropout count.
    """

    # Isolate non-dropout students (G3 != 0) for stats that exclude dropouts
    non_dropout = df[df["G3"] != 0]

    # Class average G3, computed only from non-dropout students
    class_average_g3 = round(float(non_dropout["G3"].mean()), 2)

    # Pass rate among non-dropout students (G3 >= 10)
    pass_count = int((non_dropout["G3"] >= 10).sum())
    pass_rate_percent = round((pass_count / len(non_dropout)) * 100, 2)

    # At-risk: G3 between 1 and 9 inclusive (across the full dataset)
    at_risk_count = int(((df["G3"] >= 1) & (df["G3"] <= 9)).sum())

    # Dropout: G3 == 0
    dropout_count = int((df["G3"] == 0).sum())

    return {
        "total_students": int(len(df)),
        "class_average_g3": class_average_g3,
        "pass_rate_percent": pass_rate_percent,
        "at_risk_count": at_risk_count,
        "dropout_count": dropout_count,
    }


# -----------------------------------------------------------------------
# GET /at-risk
# Returns students whose G3 is between 1 and 9 (at risk of failing),
# sorted by G3 ascending so the worst-performing students appear first.
# -----------------------------------------------------------------------
@app.get("/at-risk")
def get_at_risk_students():
    """
    Returns a list of at-risk students (G3 between 1 and 9 inclusive),
    including student_index, G1, G2, G3, and absences.
    Sorted by G3 ascending (worst first).
    """

    # Filter for at-risk students
    at_risk_df = df[(df["G3"] >= 1) & (df["G3"] <= 9)].copy()

    # Preserve original row position as student_index before sorting
    at_risk_df["student_index"] = at_risk_df.index

    # Sort by G3 ascending (worst first)
    at_risk_df = at_risk_df.sort_values(by="G3", ascending=True)

    # Select only the requested columns and convert to list of dicts
    result = at_risk_df[["student_index", "G1", "G2", "G3", "absences"]].to_dict(
        orient="records"
    )

    return result


# -----------------------------------------------------------------------
# GET /top-students
# Returns the top 5 performing students by G3, excluding dropouts,
# sorted by G3 descending.
# -----------------------------------------------------------------------
@app.get("/top-students")
def get_top_students():
    """
    Returns the top 5 students by G3 (excluding dropouts),
    including student_index, G1, G2, G3.
    Sorted by G3 descending (best first).
    """

    # Exclude dropouts (G3 == 0)
    non_dropout = df[df["G3"] != 0].copy()

    # Preserve original row position as student_index before sorting
    non_dropout["student_index"] = non_dropout.index

    # Sort by G3 descending and take the top 5
    top_5 = non_dropout.sort_values(by="G3", ascending=False).head(5)

    # Select only the requested columns and convert to list of dicts
    result = top_5[["student_index", "G1", "G2", "G3"]].to_dict(orient="records")

    return result

# -----------------------------------------------------------------------
# Pydantic model: StudentInput
# Defines and validates the input fields required for a G3 prediction.
# -----------------------------------------------------------------------
class StudentInput(BaseModel):
    G1: float = Field(
        ...,
        ge=0,
        le=20,
        description="First period grade (0-20)",
    )
    G2: float = Field(
        ...,
        ge=0,
        le=20,
        description="Second period grade (0-20)",
    )
    studytime: int = Field(
        ...,
        ge=1,
        le=4,
        description="Weekly study time level (1-4)",
    )
    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of school absences (0-100)",
    )
    failures: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of past class failures (0-4)",
    )

    # --- Custom validation error messages for each field ---
    @field_validator("G1")
    @classmethod
    def validate_g1(cls, value):
        if not (0 <= value <= 20):
            raise ValueError("G1 must be between 0 and 20")
        return value

    @field_validator("G2")
    @classmethod
    def validate_g2(cls, value):
        if not (0 <= value <= 20):
            raise ValueError("G2 must be between 0 and 20")
        return value

    @field_validator("studytime")
    @classmethod
    def validate_studytime(cls, value):
        if not (1 <= value <= 4):
            raise ValueError("studytime must be between 1 and 4")
        return value

    @field_validator("absences")
    @classmethod
    def validate_absences(cls, value):
        if not (0 <= value <= 100):
            raise ValueError("absences must be between 0 and 100")
        return value

    @field_validator("failures")
    @classmethod
    def validate_failures(cls, value):
        if not (0 <= value <= 4):
            raise ValueError("failures must be between 0 and 4")
        return value


# -----------------------------------------------------------------------
# POST /predict-result
# Accepts student input data and returns an estimated G3 grade,
# a prediction label, and a confidence level.
# -----------------------------------------------------------------------
@app.post("/predict-result")
def predict_result(student: StudentInput):
    """
    Calculates an estimated final grade (G3) from student input using
    a weighted formula, then derives a prediction label and confidence
    level from that estimate.
    """

    # --- Calculate estimated_g3 using the given weighted formula ---
    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # --- Clamp estimated_g3 to the valid grade range [0, 20] ---
    estimated_g3 = max(0, min(20, estimated_g3))

    # --- Determine prediction label based on the clamped estimate ---
    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:  # estimated_g3 >= 10
        prediction = "Pass"

    # --- Determine confidence based on G1 and G2 input values ---
    if student.G1 > 12 and student.G2 > 12:
        confidence = "High"
    elif student.G1 < 8 and student.G2 < 8:
        confidence = "High"
    else:
        confidence = "Medium"

    # --- Return the prediction results ---
    return {
        "estimated_g3": round(estimated_g3, 2),
        "prediction": prediction,
        "confidence": confidence,
    }

# -----------------------------------------------------------------------
# GET /
# Root endpoint - returns basic API info and a pointer to the docs.
# -----------------------------------------------------------------------
@app.get("/")
def root():
    """
    Root endpoint providing basic information about the API.
    """
    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0",
    }


# -----------------------------------------------------------------------
# Main block - runs the API server with uvicorn when executed directly
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)