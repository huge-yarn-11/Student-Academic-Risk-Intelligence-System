import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

def load_and_prepare_data(filepath):
    """
    Loads the UCI Student Performance (Maths.csv) dataset and engineers
    additional features used for academic risk analysis.

    Parameters
    ----------
    filepath : str
        Path to the Maths.csv file.

    Returns
    -------
    pd.DataFrame
        The original data plus engineered feature columns.
    """

    # -----------------------------------------------------------------
    # 1. Load the CSV file into a DataFrame
    # -----------------------------------------------------------------
    data_path = Path(filepath)
    if not data_path.is_absolute():
        data_path = PROJECT_DIR / data_path
    df = pd.read_csv(data_path)

    # -----------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------

    # --- Result: categorical outcome based on G3 ---
    # G3 == 0  -> "Dropout" (special case, NOT treated as a low score)
    # G3 1-9   -> "Fail"
    # G3 10-20 -> "Pass"
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:  # 10-20
            return "Pass"

    df["Result"] = df["G3"].apply(classify_result)

    # --- Percentage: final grade expressed as a percentage of 20 ---
    df["Percentage"] = (df["G3"] / 20) * 100

    # --- avg_alcohol: average of workday (Dalc) and weekend (Walc) alcohol consumption ---
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # --- parent_edu_avg: average education level of mother (Medu) and father (Fedu) ---
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # --- grade_trend: change in performance from first period (G1) to final grade (G3) ---
    # Positive value = improvement, negative value = decline
    df["grade_trend"] = df["G3"] - df["G1"]

    # --- total_support: count of "yes" responses across support-related columns ---
    # (school support, family support, paid extra classes)
    support_cols = ["schoolsup", "famsup", "paid"]
    df["total_support"] = (df[support_cols] == "yes").sum(axis=1)

    # --- risk_score: composite score combining multiple risk factors ---
    # Higher failures, more absences, and higher alcohol use increase risk;
    # more study time decreases risk.
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # --- g1_g2_avg: average of first and second period grades ---
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # -----------------------------------------------------------------
    # 3. Return the enriched DataFrame
    # -----------------------------------------------------------------
    return df

def calculate_statistics(df):
    """
    Computes summary statistics for the student dataset using NumPy,
    excluding dropouts (G3 == 0) where noted.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_and_prepare_data().

    Returns
    -------
    dict
        Dictionary containing the computed statistics.
    """

    # -----------------------------------------------------------------
    # Isolate non-dropout students (G3 != 0) for stats that should
    # exclude dropouts
    # -----------------------------------------------------------------
    non_dropout = df[df["G3"] != 0]

    # --- class_avg_g3: mean final grade among non-dropout students ---
    class_avg_g3 = np.mean(non_dropout["G3"].to_numpy())

    # --- pass_rate: % of non-dropout students who passed (G3 >= 10) ---
    passed_mask = non_dropout["G3"] >= 10
    pass_rate = (np.sum(passed_mask.to_numpy()) / len(non_dropout)) * 100

    # --- dropout_count: total students with G3 == 0 ---
    dropout_count = int(np.sum((df["G3"] == 0).to_numpy()))

    # --- at_risk_count: students with G3 between 1 and 9 inclusive ---
    at_risk_mask = (df["G3"] >= 1) & (df["G3"] <= 9)
    at_risk_count = int(np.sum(at_risk_mask.to_numpy()))

    # --- correlation_matrix: correlation between G1, G2, G3 (non-dropouts only) ---
    # np.corrcoef expects variables as rows, so we stack the three columns
    grades_array = np.vstack(
        [
            non_dropout["G1"].to_numpy(),
            non_dropout["G2"].to_numpy(),
            non_dropout["G3"].to_numpy(),
        ]
    )
    correlation_matrix = np.corrcoef(grades_array)

    # -----------------------------------------------------------------
    # Package results into a dictionary
    # -----------------------------------------------------------------
    stats = {
        "total_students": len(df),
        "class_avg_g3": class_avg_g3,
        "pass_rate": pass_rate,
        "dropout_count": dropout_count,
        "at_risk_count": at_risk_count,
        "correlation_matrix": correlation_matrix,
    }

    return stats


def generate_static_charts(df, output_dir=None):
    """
    Generates and saves two static charts summarizing the dataset:
    1. Bar chart of average G3 by study time level
    2. Pie chart of Pass/Fail/Dropout distribution

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_and_prepare_data().
    """

    # -----------------------------------------------------------------
    # Ensure the output folder exists before saving anything
    # -----------------------------------------------------------------
    output_path = Path(output_dir) if output_dir else PROJECT_DIR / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Chart 1: Bar chart - Average G3 by Study Time
    # ===================================================================

    # Group by studytime (1-4) and compute mean G3 for each level
    avg_g3_by_studytime = df.groupby("studytime")["G3"].mean()

    plt.figure(figsize=(8, 6))
    plt.bar(avg_g3_by_studytime.index, avg_g3_by_studytime.values, color="steelblue")

    # Titles and axis labels as specified
    plt.title("Average G3 by Study Time")
    plt.xlabel("Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)")
    plt.ylabel("Average G3")

    # Ensure all 4 studytime levels show as ticks on the x-axis
    plt.xticks(avg_g3_by_studytime.index)

    # Save to output folder and close the figure to free memory
    plt.savefig(output_path / "avg_g3_by_studytime.png")
    plt.close()

    # ===================================================================
    # Chart 2: Pie chart - Result Distribution (Pass / Fail / Dropout)
    # ===================================================================

    # Count how many students fall into each Result category
    result_counts = df["Result"].value_counts()

    plt.figure(figsize=(7, 7))
    plt.pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%",  # show percentage on each slice
        startangle=90,
    )
    plt.title("Student Result Distribution")

    # Save to output folder and close the figure to free memory
    plt.savefig(output_path / "pass_fail_dropout_pie.png")
    plt.close()

def generate_interactive_charts(df):
    """
    Generates two interactive Plotly charts:
    1. Scatter plot of studytime vs G3, colored by Result
    2. Bar chart of average G3 grouped by internet access

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_and_prepare_data().
    """

    # ===================================================================
    # Chart 1: Scatter plot - Study Time vs Final Grade (G3)
    # ===================================================================

    # Map each Result category to a specific color as requested
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
        hover_data=["absences", "G1", "G2"],  # extra info shown on hover
        title="Study Time vs Final Grade (G3)",
    )

    # Display the scatter plot
    scatter_fig.show()

    # ===================================================================
    # Chart 2: Bar chart - Average G3 by Internet Access
    # ===================================================================

    # Group by internet access (yes/no) and compute the average G3
    internet_avg_g3 = (
        df.groupby("internet")["G3"].mean().reset_index()
    )

    bar_fig = px.bar(
        internet_avg_g3,
        x="internet",
        y="G3",
        color="internet",  # color bars by internet access group
        title="Average G3 by Internet Access",
    )

    # Display the bar chart
    bar_fig.show()

def print_summary(stats):
    """
    Prints a clean, formatted summary of the analysis results.

    Parameters
    ----------
    stats : dict
        Dictionary returned by calculate_statistics().
    """

    # Header
    print("=" * 48)
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print("ANALYSIS SUMMARY")
    print("=" * 48)

    # Body rows, formatted with fixed-width labels and aligned values
    print(f"{'Total Students':<22} : {stats['total_students']}")
    print(f"{'Class Average G3':<22} : {stats['class_avg_g3']:.2f}")
    print(f"{'Pass Rate':<22} : {stats['pass_rate']:.2f}%")
    print(f"{'At-Risk Count':<22} : {stats['at_risk_count']}")
    print(f"{'Dropout Count':<22} : {stats['dropout_count']}")

    # Footer
    print("=" * 48)


# =======================================================================
# Main block - runs the full pipeline end-to-end
# =======================================================================
if __name__ == "__main__":
    # Step 1: Load data and engineer features
    df = load_and_prepare_data("data/Maths.csv")

    # Step 2: Compute summary statistics
    stats = calculate_statistics(df)

    # Step 3: Generate and save static (Matplotlib) charts
    generate_static_charts(df)

    # Step 4: Generate and display interactive (Plotly) charts
    generate_interactive_charts(df)

    # Step 5: Print the formatted summary table
    print_summary(stats)

    # Final confirmation message
    print("Analysis complete. Charts saved to output/ folder")