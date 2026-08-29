# Student Academic Risk Intelligence System
 
A data analytics system for identifying academically at-risk students using the UCI **Student Performance (Maths)** dataset. The project engineers risk-related features from raw student records and exposes them through three complementary interfaces: a batch analysis script, a REST API, and an interactive dashboard.
 
## Overview
 
Schools and educators often need a way to flag students who are likely to fail or drop out before it's too late. This project analyzes student demographic, social, and academic data to:
 
- Classify each student's outcome as **Pass**, **Fail**, or **Dropout**
- Compute a composite **risk score** from failures, absences, alcohol use, and study time
- Surface **at-risk** and **top-performing** students
- Estimate a student's likely final grade from partial data (early grades, study habits, absences)
## Features
 
- **Feature engineering** — derives `Result`, `Percentage`, `avg_alcohol`, `parent_edu_avg`, `grade_trend`, `total_support`, `risk_score`, and `g1_g2_avg` from the raw dataset
- **Static & interactive charts** — Matplotlib bar/pie charts saved to disk, plus Plotly scatter/bar charts
- **REST API** (FastAPI) — summary statistics, at-risk students, top students, and a grade prediction endpoint
- **Interactive dashboard** (Streamlit) — KPI cards, filterable data tables, and live charts
- **Command-line analysis** — a standalone script that prints a full summary report and saves charts
## Project Structure
 
```
Student-Academic-Risk-Intelligence-System/
├── analysis.py         # Standalone data analysis pipeline (stats + charts)
├── app.py               # Streamlit dashboard
├── main.py               # FastAPI application
├── requirements.txt       # Python dependencies
├── data/
│   └── maths.csv         # UCI Student Performance dataset (Maths course)
└── output/                # Generated charts (created on running analysis.py)
```
 
## Dataset
 
The system uses the **UCI Student Performance Dataset (Maths)**, containing 33 columns covering demographics (age, sex, address), family background (parents' education/jobs, family support), lifestyle (study time, alcohol consumption, free time), and academic history (`G1`, `G2`, `G3` — grades from three periods, each 0–20).
 
**Result classification** is based on the final grade `G3`:
| G3 Range | Result |
|---|---|
| 0 | Dropout |
| 1–9 | Fail |
| 10–20 | Pass |
 
## Installation
 
```bash
# Clone the repository
git clone <repository-url>
cd Student-Academic-Risk-Intelligence-System
 
# Install dependencies
pip install -r requirements.txt
```
 
**Requirements:** Python 3.8+, pandas, numpy, matplotlib, plotly, streamlit, fastapi, uvicorn, pydantic
 
## Usage
 
### 1. Run the batch analysis
 
Generates summary statistics, saves static charts to `output/`, and shows interactive Plotly charts.
 
```bash
python analysis.py
```
 
### 2. Launch the API server
 
```bash
python main.py
# or
uvicorn main:app --reload
```
 
The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.
 
**Endpoints:**
 
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/summary` | Total students, class average, pass rate, at-risk & dropout counts |
| GET | `/at-risk` | Students with G3 between 1–9, sorted worst first |
| GET | `/top-students` | Top 5 performing students (excluding dropouts) |
| POST | `/predict-result` | Estimates final grade from `G1`, `G2`, `studytime`, `absences`, `failures` |
 
**Example request:**
```bash
curl -X POST http://localhost:8000/predict-result \
  -H "Content-Type: application/json" \
  -d '{"G1": 12, "G2": 13, "studytime": 2, "absences": 4, "failures": 0}'
```
 
### 3. Launch the dashboard
 
```bash
streamlit run app.py
```
 
Opens an interactive dashboard with KPI cards, performance charts (study time vs. grade, internet access vs. grade), a filterable student table, and a dedicated at-risk students view.
 
## Key Engineered Features
 
| Feature | Description |
|---|---|
| `Result` | Pass / Fail / Dropout classification based on G3 |
| `Percentage` | Final grade as a percentage of 20 |
| `avg_alcohol` | Average of workday (`Dalc`) and weekend (`Walc`) alcohol consumption |
| `parent_edu_avg` | Average of mother's and father's education level |
| `grade_trend` | Change in performance from G1 to G3 |
| `total_support` | Count of "yes" responses across school/family support and paid classes |
| `risk_score` | Composite score: `(failures × 2) + (absences / 10) + avg_alcohol − studytime` |
| `g1_g2_avg` | Average of first and second period grades |
 
## License
 
This project uses the publicly available UCI Student Performance Dataset for educational and analytical purposes.
