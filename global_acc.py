import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for LLM predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    return mae, mse, rmse, r2, bias

def process_results_folder(folder_path, output_file):
    """Process all Excel files in the folder and compute metrics for each model."""
    all_results = []
    
    # Loop through all Excel files in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".xlsx"):
                df = pd.read_excel(file_path, engine="openpyxl")
            elif file.endswith(".xls"):
                df = pd.read_excel(file_path, engine="xlrd")
            else:
                print(f"Skipping {file}, unsupported file format.")
                continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
            
        if "Truth" not in df.columns or "number_of_people" not in df.columns:
            print(f"Skipping {file}, missing required columns.")
            continue
        
        filename_column = df["filename"] if "filename" in df.columns else None
        truth = df["Truth"]
        y_pred = df["number_of_people"]
        
        if y_pred.isnull().sum() > 0:
            print(f"Warning: 'number_of_people' column in file {file} contains NaN values. Skipping...")
            continue  # Skip if NaN values are present
        print(file)
        mae, mse, rmse, r2, bias = calculate_metrics(truth, y_pred)
        
        all_results.append({
            "File": file,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2,
            "Bias": bias
        })
    
    # Save results to an Excel file
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(output_file, index=False, engine="openpyxl")
        print(f"Metrics saved to {output_file}")
    else:
        print("No valid files processed.")

# Usage
results_folder = "results"  # Change this to your folder path
output_metrics_file = "LLM_NBC_Evaluation.xlsx"
process_results_folder(results_folder, output_metrics_file)
