import pandas as pd
from faker import Faker

# Load your dataset
df = pd.read_csv("data.csv")  # Ensure "data.csv" is in the same folder

# Remove unwanted columns
columns_to_drop = [
    "DailyRate", "EmployeeCount", "HourlyRate", 
    "MonthlyRate", "Over18", "StandardHours",
    "JobLevel", "TotalWorkingYears", "PercentSalaryHike", 
    "YearsInCurrentRole", "YearsWithCurrManager"
]
df.drop(columns=columns_to_drop, inplace=True)

# Add realistic names based on gender
fake = Faker()
def generate_name(gender):
    if gender == "Female":
        return fake.first_name_female() + " " + fake.last_name()
    else:
        return fake.first_name_male() + " " + fake.last_name()

df["Name"] = df["Gender"].apply(generate_name)

# Reorder columns to place "Name" first
df.insert(0, "Name", df.pop("Name"))

# Save to new CSV
df.to_csv("processed_data.csv", index=False)
print("Processing complete! Check 'processed_employee_data.csv'.")
