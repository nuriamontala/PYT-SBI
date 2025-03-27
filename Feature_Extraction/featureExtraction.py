import os
import pandas as pd
import numpy as np
import time
import traceback
from sequenceProfiles import *
from parsePDB import *
from physchemFeatures import *
from structuralFeatures import *

def merge_data_to_csv(physicochemical_data, structural_data, pssm_data, hmm_data, binding_data, output_file):
    final_data = {}
    mismatch_found = False
    with open("errores.log", "a") as error_log:
        for key in binding_data.keys():
            if key in physicochemical_data and key in pssm_data and key in hmm_data and key in structural_data:              
                final_data[key] = physicochemical_data[key] + structural_data[key] + pssm_data[key] + hmm_data[key] + binding_data[key]
            else:
                error_log.write(f">{output_file} Mismatch for structural key {key}\n")
                mismatch_found=True

    if not final_data:
        print("--------------No data to write in the CSV.")
        return
    
    if mismatch_found:
        print("--------------Non coincident residues were found. See error.log")

    # Convert to DataFrame
    residue_ids = np.array(list(final_data.keys()), dtype=object)
    features = np.array(list(final_data.values()))
    df = pd.DataFrame(features, index=residue_ids, columns=column_names)
    # Save as CSV
    df.to_csv(output_file, index_label="Residue_ID")
    print(f"--------------File saved: {output_file}")



def main():
    with open("errores.log", "w") as f:
        f.write("")

    pdb_directory="../files/pdb/original_files"
    pdb_site_directory="../files/pdb/protwithsitefinal"
    fasta_directory="../files/fasta"
    psiblast_directory="../files/psiblast"
    jackhmmr_directory="../files/jackhmmr"
    features_directory = "../files/features"
    database_psiblast = "../database/uniprot_sprot_db"
    database_jackhmmer = "../database/uniprot_sprot.fasta"

    with open("pdbs_marti.txt") as f:
        pdb_filtered = [line.strip() for line in f]

    # pdb_filtered = ["1a4g"]
    for pdb_id in pdb_filtered[2001:]:
        try:
            # Start timer
            start_time = time.time() 

            # Extract fasta for each chain
            chains=pdb2fasta(pdb_id,  pdb_directory, fasta_directory)
            offset=get_offset_from_pdb(pdb_id, pdb_directory)
            # Get physicochemical and structural information from the whole pdb
            physicochemical_data=get_physicochemical_features(pdb_id, pdb_directory)
            structural_data=get_structural_features(pdb_id, pdb_directory)
            # Run PSI-BLAST for each chain
            run_psiblast(chains, database_psiblast, fasta_directory, psiblast_directory)
            pssm_data=parse_pssm(chains, psiblast_directory, offset)
            # Run jackhmmr for each chain
            run_jackhmmer(chains, database_jackhmmer, fasta_directory, jackhmmr_directory,num_iterations=3)
            hmm_data=parse_hmm(chains, jackhmmr_directory, offset)
            # Get binding site label
            binding_data=get_binding_sites(pdb_id, pdb_site_directory)
            # Print keys
            # print([k for k in structural_data.keys() if k[1]=="B"])
            # print([k for k in hmm_data.keys() if k[1]=="B"])
            # print([k for k in binding_data.keys() if k[1]=="B"])
            # Merge all the features in one numpy matrix
            merge_data_to_csv(physicochemical_data, structural_data, pssm_data, hmm_data, binding_data, f"{features_directory}/{pdb_id}.csv")
          
            # Remove files from fasta, psiblast and jackhmmr directories
            for folder in [fasta_directory, psiblast_directory, jackhmmr_directory]:
                if os.path.exists(folder) and os.path.isdir(folder):  # Verifica que la carpeta exista
                    for file in os.listdir(folder):
                        file_path = os.path.join(folder, file)
                        os.remove(file_path)

            # Print execution time of iteration        
            end_time = time.time() 
            elapsed_time = end_time - start_time  
            print(f"Total time of execution: {elapsed_time:.2f} seconds")
        except Exception as e:
            with open("errors.log", "a") as f:
                f.write(f"{pdb_id}: {e}\n")
                f.write(traceback.format_exc() + "\n")

if __name__ == "__main__":
    main()
