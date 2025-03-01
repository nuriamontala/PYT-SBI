import os
import pandas as pd

# Define directories
input_dir = "../files/features_nuria"  # Change to the directory containing your CSV files
output_dir = "../files/normalized_features"  # Change to where you want to save the normalized files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all CSV files in the input directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# Initialize list to store selected feature data for global min-max calculation
all_data = []

# Step 1: Read all files and collect selected feature data
for file in csv_files:
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # Select columns: 2 to 22 and 43 to penultimate
    selected_columns = list(df.columns[1:22]) + list(df.columns[42:-1])
    all_data.append(df[selected_columns])  # Store selected features for min-max computation

# Step 2: Compute per-column global min and max across all files
combined_data = pd.concat(all_data, ignore_index=True)  # Combine all selected feature data
global_min = combined_data.min()  # Min per column
global_max = combined_data.max()  # Max per column

# Step 3: Normalize each file using per-column global min-max
for file in csv_files:
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # Apply min-max normalization only to selected columns
    df[selected_columns] = (df[selected_columns] - global_min) / (global_max - global_min)

    # Save normalized file
    output_path = os.path.join(output_dir, file)
    df.to_csv(output_path, index=False)

print(f"Normalization complete. Normalized files saved to {output_dir}.")
