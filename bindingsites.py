import os
import re
import numpy as np

# Amino acid properties dictionary (without Mean Accessibility)
AA_PROPERTIES = {
    "ALA": [1.24, 0.62, 0.38, 6.00],
    "ARG": [2.74, -2.53, 0.89, 10.76],
    "ASN": [2.14, -0.78, 0.48, 5.41],
    "ASP": [2.16, -0.90, 0.49, 2.85],
    "CYS": [1.50, 0.29, 0.54, 5.07],
    "GLN": [2.17, -0.85, 0.51, 5.65],
    "GLU": [2.18, -0.74, 0.52, 3.15],
    "GLY": [0.00, 0.48, 0.00, 6.06],
    "HIS": [2.48, -0.40, 0.69, 7.60],
    "ILE": [3.08, 1.25, 0.76, 6.05],
    "LEU": [2.80, 1.22, 0.74, 6.01],
    "LYS": [2.90, -2.25, 0.84, 9.74],
    "MET": [2.67, 1.02, 0.72, 5.74],
    "PHE": [2.58, 1.47, 0.78, 5.48],
    "PRO": [1.95, 0.09, 0.64, 6.30],
    "SER": [1.31, -0.28, 0.41, 5.68],
    "THR": [1.50, -0.18, 0.44, 5.60],
    "TRP": [3.07, 1.45, 0.81, 5.89],
    "TYR": [2.67, 0.94, 0.76, 5.64],
    "VAL": [2.50, 1.08, 0.71, 6.00],
}

def process_pdb_file(input_file, output_file):
    """Processes a PDB file and extracts unique residues with binding site labels, properties, and Mean B-factor."""
    residue_dict = {}  # { (Residue_Number, Chain) : [Binding_Site, Residue_Type, Properties, B-factors] }

    with open(input_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "SITE")):  # Read only ATOM and SITE lines
                residue_number = line[22:26].strip()  # Residue number
                chain_id = line[21].strip()  # Chain ID
                residue_type = line[17:20].strip()  # Residue type

                try:
                    b_factor = float(line[60:66].strip())  # Extract B-factor
                except ValueError:
                    b_factor = 0.0  # Default if invalid

                key = (residue_number, chain_id)

                # Retrieve amino acid properties, default to zeros if unknown
                properties = AA_PROPERTIES.get(residue_type, [0.0] * 4)

                # If it's a SITE residue, mark as binding site (1)
                if key not in residue_dict:
                    residue_dict[key] = [0, residue_type] + properties + [[]]
                if line.startswith("SITE"):
                    residue_dict[key][0] = 1  # Mark as binding site
                residue_dict[key][-1].append(b_factor)  # Store B-factor

    # Compute mean B-factor for each residue (excluding zero values)
    for key in residue_dict:
        b_factors = residue_dict[key][-1]  # Extract B-factor list
        non_zero_b_factors = [b for b in b_factors if b > 0]  # Exclude zero values
        residue_dict[key][-1] = float(np.mean(non_zero_b_factors)) if non_zero_b_factors else 0.0  # Ensure float, not list

    # Sorting function
    def sorting_key(x):
        match = re.match(r"(\d+)([A-Z]?)", x[0][0])  # Extract number and optional letter
        if match:
            num = int(match.group(1))  # Numeric part
            ins_code = match.group(2)  # Insertion code (if any)
        else:
            num = float('inf')  # Assign a high number to place unexpected values at the end
            ins_code = ""
        return x[0][1], num, ins_code  # Sort by (Chain, Numeric Residue Number, Insertion Code)

    # Save output as TSV
    with open(output_file, "w") as out_f:
        out_f.write("Residue_Number\tChain\tResidue\tBinding_Site\tVolume\tPolarity\tHydropathy\tpKa\tMean_BFactor\n")

        for (res_num, chain), values in sorted(residue_dict.items(), key=sorting_key):
            # Debugging: Ensure correct number of elements
            if len(values) != 7:  # 1 Binding_Site + 1 Residue + 4 Properties + 1 Mean B-Factor
                print(f"ERROR: Unexpected value count for {res_num} {chain}: {values}")

            out_f.write(f"{res_num}\t{chain}\t{values[1]}\t{values[0]}\t{'\t'.join(map(str, values[2:-1]))}\t{values[-1]:.2f}\n")


def process_pdb_files(input_folder, output_folder):
    """Processes all PDB files in a folder."""
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdb"):
            pdb_code = filename.split("_")[0]  # Extract PDB code
            pdb_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, f"{pdb_code}_site.tsv")  # Change extension to .tsv

            process_pdb_file(pdb_file_path, output_file_path)
            print(f"Processed: {filename} -> {output_file_path}")

# Set input and output folder paths
input_folder = "protwithsitefinal"
output_folder = "binding_sites"

# Run the processing
process_pdb_files(input_folder, output_folder)
