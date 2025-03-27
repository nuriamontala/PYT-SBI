import os
import pandas as pd

# Define directories
input_dir = "../files/features"  # Change to the directory containing your CSV files
output_dir = "../files/normalized_features"  # Change to where you want to save the normalized files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all CSV files in the input directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# Initialize list to store selected feature data for global min-max calculation
all_data = []
common_columns = None  # This will store consistent columns across all files

# Step 1: Read all files and collect selected feature data
for file in csv_files:
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # Select columns: 2 to 22 and 43 to penultimate
    selected_columns = list(df.columns[1:22]) + list(df.columns[42:-1])

    if common_columns is None:
        common_columns = set(selected_columns)
    else:
        common_columns.intersection_update(selected_columns)

# Convert set to list and ensure consistent order
common_columns = sorted(list(common_columns))

# Collect data for min-max normalization
for file in csv_files:
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # Filter dataframe to only include common columns
    valid_columns = [col for col in common_columns if col in df.columns]
    
    if not valid_columns:
        print(f"Skipping {file} - No valid columns found.")
        continue  # Skip this file

    all_data.append(df[valid_columns])

# Step 2: Compute per-column global min and max across all files
combined_data = pd.concat(all_data, ignore_index=True)  # Combine all selected feature data
global_min = combined_data.min()  # Min per column
global_max = combined_data.max()  # Max per column

# Handle potential division by zero
global_range = global_max - global_min
global_range[global_range == 0] = 1  # Avoid division by zero

# Save global min and max values to a CSV file
min_max_df = pd.DataFrame({'Min': global_min, 'Max': global_max})
min_max_csv_path = os.path.join(output_dir, "global_min_max.csv")
min_max_df.reset_index().to_csv(min_max_csv_path, index=False, sep='\t')

# Step 3: Normalize each file using per-column global min-max
for file in csv_files:
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # Ensure valid columns exist in the DataFrame
    valid_columns = [col for col in common_columns if col in df.columns]
    
    if not valid_columns:
        print(f"Skipping {file} - No valid columns found.")
        continue  # Skip this file

    # Select only numeric columns for normalization
    df_numeric = df[valid_columns].select_dtypes(include=['number'])

    # Ensure global min/max only contain relevant columns
    min_values = global_min[valid_columns]
    max_values = global_max[valid_columns]

    # Apply min-max normalization
    df[valid_columns] = (df_numeric - min_values) / (max_values - min_values)

    # Save normalized file
    output_path = os.path.join(output_dir, file)
    df.to_csv(output_path, index=False, sep='\t')  # Normalized files

print(f"Normalization complete. Normalized files saved to {output_dir}.")
print(f"Global min-max values saved to {min_max_csv_path}.")
