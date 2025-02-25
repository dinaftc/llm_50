import os
import pandas as pd

# Define file paths
csv_folder = "python_ollama_code"
results_folder = "results"  # Folder to store results
ground_truth_file = "ground_truth/list_50.xlsx"

# Create results folder if it does not exist
os.makedirs(results_folder, exist_ok=True)

print(f"Processing CSV files from: {csv_folder}")
print(f"Ground truth file: {ground_truth_file}")

# Load ground truth Excel file
gt_df = pd.read_excel(ground_truth_file)
print(gt_df)

# Ensure proper column names
gt_df.columns = ["filename", "Truth"]
print(f"Ground truth columns: {gt_df.columns}")

# Loop through each CSV file in python_ollama_code
for csv_file in os.listdir(csv_folder):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(csv_folder, csv_file)

        # Load CSV file
        pred_df = pd.read_csv(csv_path)
        print(f"Processing file: {csv_file}")
        print(pred_df)

        # Ensure proper column names
        pred_df.columns = ["filename", "number_of_people"]

        # Merge with ground truth data
        merged_df = pred_df.merge(gt_df, on="filename", how="left")

        # Compute accuracy percentage
        def compute_accuracy(row):
            if pd.notnull(row["Truth"]) and row["Truth"] != 0:
                return round((row["number_of_people"] / row["Truth"]) * 100, 2)
            return "N/A"

        merged_df["Accuracy (%)"] = merged_df.apply(compute_accuracy, axis=1)

        # Define the output file path inside the "results" folder
        results_filename = os.path.join(results_folder, f"results_{csv_file.replace('.csv', '.xlsx')}")

        # Save results to an Excel file
        merged_df.to_excel(results_filename, index=False)

        print(f"Results saved: {results_filename}")
