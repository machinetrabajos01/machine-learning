import pandas as pd

# Load the final OULAD-based dataset
file_path = "data/student_performance_1200.csv"

student_data = pd.read_csv(file_path)

print("=== DATASET INFORMATION ===")
print(f"Number of records: {len(student_data)}")
print(f"Number of columns: {len(student_data.columns)}")

print("\n=== COLUMNS ===")
print(student_data.columns.tolist())

print("\n=== DATA TYPES ===")
print(student_data.dtypes)

print("\n=== MISSING VALUES ===")
print(student_data.isnull().sum())

print("\n=== DUPLICATE RECORDS ===")
print(student_data.duplicated().sum())

print("\n=== DESCRIPTIVE STATISTICS ===")
print(student_data.describe())

print("\n=== FIRST 10 RECORDS ===")
print(student_data.head(10).to_string(index=False))

print("\n=== LAST 10 RECORDS ===")
print(student_data.tail(10).to_string(index=False))