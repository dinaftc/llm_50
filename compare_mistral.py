import os
import pandas as pd

def load_mistral_data(folder_path):
    """Load all CSV files in the given folder and merge them into a single DataFrame."""
    mistral_data = {}
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, names=["filename", file.replace(".csv", "")])
            df[file.replace(".csv", "")] = pd.to_numeric(df[file.replace(".csv", "")], errors='coerce')
            mistral_data[file] = df
    
    # Merge all dataframes on filename
    merged_df = None
    for df in mistral_data.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="filename", how="outer")
    
    return merged_df

def load_ground_truth(ground_truth_file):
    """Load the ground truth data from an Excel file."""
    df_truth = pd.read_excel(ground_truth_file, sheet_name=None)
    sheet_name = list(df_truth.keys())[0]  # Assuming data is in the first sheet
    df_truth = df_truth[sheet_name]
    df_truth.columns = ["filename", "Truth"]
    df_truth["Truth"] = pd.to_numeric(df_truth["Truth"], errors='coerce')
    return df_truth

def compare_results(mistral_folder, ground_truth_file, output_file):
    """Compare the results from Mistral CSV files with the ground truth and save the output."""
    mistral_df = load_mistral_data(mistral_folder)
    truth_df = load_ground_truth(ground_truth_file)
    
    # Merge the mistral results with ground truth
    final_df = pd.merge(mistral_df, truth_df, on="filename", how="outer")
    
    # Compute Mean Absolute Error (MAE) for each file
    error_results = {}
    for col in final_df.columns[1:-1]:  # Exclude filename and Truth columns
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        final_df[f"error_{col}"] = abs(final_df[col] - final_df["Truth"])
        error_results[col] = final_df[f"error_{col}"].mean()
    
    # Identify the most accurate file
    most_accurate_file = min(error_results, key=error_results.get)
    print(f"The most accurate file is {most_accurate_file} with MAE = {error_results[most_accurate_file]:.2f}")
    
    # Save the results to an output CSV or Excel file
    final_df.to_excel(output_file, index=False)
    print(f"Comparison saved to {output_file}")

# Example usage
mistral_folder = "mistral_params"  # Change this to your actual folder path
ground_truth_file = "ground_truth/list_50.xlsx"  # Change this to your actual file path
output_file = "comparison_results.xlsx"

compare_results(mistral_folder, ground_truth_file, output_file)