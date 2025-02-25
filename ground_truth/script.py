import os
import pandas as pd

# Folder path
folder_path = "../data"

# Get list of files
files = os.listdir(folder_path)

# Create a DataFrame
df = pd.DataFrame(files, columns=["Filename"])

# Save to Excel and CSV
df.to_excel("list_50.xlsx", index=False)
df.to_csv("list_50.csv", index=False)

print("Excel and CSV files have been created successfully!")
