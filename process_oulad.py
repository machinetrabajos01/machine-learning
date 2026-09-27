import pandas as pd
from pathlib import Path

# Define the OULAD data directory
OULAD_DIR = Path("oulad")

# Define the output directory
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load OULAD files
student_info = pd.read_csv(OULAD_DIR / "studentInfo.csv")
student_assessment = pd.read_csv(OULAD_DIR / "studentAssessment.csv")
student_vle = pd.read_csv(OULAD_DIR / "studentVle.csv")

print("OULAD files loaded successfully.")
print(f"studentInfo records: {len(student_info)}")
print(f"studentAssessment records: {len(student_assessment)}")
print(f"studentVle records: {len(student_vle)}")

# Convert assessment scores to numeric values
student_assessment["score"] = pd.to_numeric(
    student_assessment["score"],
    errors="coerce"
)

# Convert VLE clicks to numeric values
student_vle["sum_click"] = pd.to_numeric(
    student_vle["sum_click"],
    errors="coerce"
)

# Remove invalid assessment scores
student_assessment = student_assessment.dropna(
    subset=["id_student", "score"]
)

# Remove invalid VLE click values
student_vle = student_vle.dropna(
    subset=["id_student", "sum_click"]
)

# Calculate the average assessment score for each student
assessment_summary = (
    student_assessment
    .groupby("id_student")["score"]
    .mean()
    .reset_index()
)

assessment_summary.rename(
    columns={"score": "Average_Assessment_Score"},
    inplace=True
)

# Calculate total VLE clicks for each student
vle_summary = (
    student_vle
    .groupby("id_student")["sum_click"]
    .sum()
    .reset_index()
)

vle_summary.rename(
    columns={"sum_click": "Total_VLE_Clicks"},
    inplace=True
)

# Combine assessment and VLE information
student_performance = pd.merge(
    assessment_summary,
    vle_summary,
    on="id_student",
    how="inner"
)

# Remove students with missing values
student_performance.dropna(
    subset=[
        "Average_Assessment_Score",
        "Total_VLE_Clicks"
    ],
    inplace=True
)

# Select a reproducible random sample of 1,200 students
student_performance = student_performance.sample(
    n=1200,
    random_state=42
).copy()

# Rename the student identifier
student_performance.rename(
    columns={"id_student": "Student_ID"},
    inplace=True
)

# Round numerical values
student_performance["Average_Assessment_Score"] = (
    student_performance["Average_Assessment_Score"].round(2)
)

student_performance["Total_VLE_Clicks"] = (
    student_performance["Total_VLE_Clicks"].round().astype(int)
)

# Save the final dataset
output_path = OUTPUT_DIR / "student_performance_1200.csv"

student_performance.to_csv(
    output_path,
    index=False
)

print("\nFinal dataset created successfully.")
print(f"Output file: {output_path}")
print(f"Final records: {len(student_performance)}")

print("\nFinal columns:")
print(student_performance.columns.tolist())

print("\nFirst 10 records:")
print(student_performance.head(10).to_string(index=False))