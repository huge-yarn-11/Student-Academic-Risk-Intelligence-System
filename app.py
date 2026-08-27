# =======================================================================
# app.py - Student Academic Risk Intelligence System (Streamlit Dashboard)
# =======================================================================

# --- Imports ---
import streamlit as st
import pandas as pd
import plotly.express as px


# -----------------------------------------------------------------------
# Page configuration - must be the first Streamlit command in the script
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide",
    page_icon="🎓",
)


# -----------------------------------------------------------------------
# load_data: loads the CSV and applies the same feature engineering
# used in analysis.py / main.py, so the dashboard stays consistent
# with the rest of the system.
# -----------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Loads Maths.csv from the data/ folder and engineers the same
    additional features used elsewhere in the system.

    Returns
    -------
    pd.DataFrame
        The original data plus engineered feature columns.
    """

    # Load the raw CSV file
    df = pd.read_csv("./data/maths.csv")

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


# Load the prepared dataset
df = load_data()


# -----------------------------------------------------------------------
# Main page title
# -----------------------------------------------------------------------
st.title("🎓 Student Academic Risk Intelligence System")


# -----------------------------------------------------------------------
# KPI metric cards - 4 cards shown in a single row
# -----------------------------------------------------------------------

# Isolate non-dropout students (G3 != 0) for stats that exclude dropouts
non_dropout = df[df["G3"] != 0]

# Card 1 value: total number of students in the dataset
total_students = len(df)

# Card 2 value: class average G3 among non-dropout students
class_avg_g3 = round(non_dropout["G3"].mean(), 2)

# Card 3 value: pass rate (%) among non-dropout students (G3 >= 10)
pass_count = (non_dropout["G3"] >= 10).sum()
pass_rate_percent = round((pass_count / len(non_dropout)) * 100, 1)

# Card 4 value: at-risk count (G3 between 1 and 9 inclusive)
at_risk_count = ((df["G3"] >= 1) & (df["G3"] <= 9)).sum()

# Create 4 equal-width columns for the KPI cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Students", value=total_students)

with col2:
    st.metric(label="Class Average G3", value=class_avg_g3)

with col3:
    st.metric(label="Pass Rate %", value=f"{pass_rate_percent}%")

with col4:
    st.metric(label="At-Risk Count", value=at_risk_count)

#prompt 11

# -----------------------------------------------------------------------
# Section: Performance Charts - two interactive Plotly charts side by side
# -----------------------------------------------------------------------
st.subheader("📊 Performance Charts")

# Create 2 equal-width columns to hold the charts side by side
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # --- Scatter plot: Study Time vs Final Grade (G3) ---
    # Color-coded by Result, with extra info shown on hover
    result_color_map = {
        "Pass": "green",
        "Fail": "red",
        "Dropout": "grey",
    }

    scatter_fig = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        color_discrete_map=result_color_map,
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade",
    )

    st.plotly_chart(scatter_fig, use_container_width=True)

with chart_col2:
    # --- Bar chart: Average G3 by Internet Access ---
    # Group by internet access (yes/no) and compute mean G3 per group
    internet_avg_g3 = df.groupby("internet")["G3"].mean().reset_index()

    bar_fig = px.bar(
        internet_avg_g3,
        x="internet",
        y="G3",
        color="internet",
        title="Average G3 by Internet Access",
    )

    st.plotly_chart(bar_fig, use_container_width=True)

# prompt 12
# -----------------------------------------------------------------------
# Section: Student Analysis Table - filterable by Result
# -----------------------------------------------------------------------
st.subheader("🚨 Student Analysis Table")

# Dropdown to filter the table by Result category
result_filter = st.selectbox(
    "Filter by Result",
    options=["All", "Pass", "Fail", "Dropout"],
)

# Apply the filter: "All" shows every student, otherwise filter by Result
if result_filter == "All":
    filtered_df = df
else:
    filtered_df = df[df["Result"] == result_filter]

# Display the filtered table with only the requested columns
table_cols = [
    "G1",
    "G2",
    "G3",
    "Result",
    "Percentage",
    "absences",
    "studytime",
    "failures",
    "risk_score",
]
st.dataframe(filtered_df[table_cols])


# -----------------------------------------------------------------------
# Section: At-Risk Students - G3 between 1 and 9, sorted worst first
# -----------------------------------------------------------------------
st.subheader("⚠️ At-Risk Students")

# Filter for at-risk students (G3 between 1 and 9 inclusive)
at_risk_df = df[(df["G3"] >= 1) & (df["G3"] <= 9)]

# Sort by G3 ascending so worst-performing students appear first
at_risk_df = at_risk_df.sort_values(by="G3", ascending=True)

# Display only the requested columns
at_risk_cols = ["G1", "G2", "G3", "absences", "studytime", "failures"]
st.dataframe(at_risk_df[at_risk_cols])

# Show the total count of at-risk students
st.write(f"Total at-risk students: {len(at_risk_df)}")
