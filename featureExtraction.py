import os
import subprocess
from Bio.PDB import PDBParser, DSSP, HSExposureCB, NeighborSearch
import glob
import pandas as pd
import numpy as np
import shutil
import time
import traceback
import multiprocessing as mp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pdb2fasta(pdb_id, pdb_directory, fasta_directory, res_dict, min_length=30, max_chains=6):
    """Extract protein sequence from a PDB file and save it as FASTA in the fasta directory."""
   
    os.makedirs(fasta_directory, exist_ok=True)

    pdb_file = os.path.join(pdb_directory, f"{pdb_id}.pdb")
    
    # Diccionario para almacenar las secuencias
    chain_dict = {}

    # Leer SEQRES del PDB
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("SEQRES"):
                parts = line.split()
                chain_id=pdb_id+"_"+parts[2]
                # Convertir residuos de tres letras a una letra
                seq = [res_dict.get(res, "X") for res in parts[4:]]  
                if chain_id in chain_dict:
                    chain_dict[chain_id] += "".join(seq)
                else:
                    chain_dict[chain_id] = "".join(seq)

    num_chains = len(chain_dict)
    if num_chains > max_chains:
        raise Exception(f"Skipping PDB. File has {num_chains} chains (limit: {max_chains})")
    # Filtwer short chains and save in FASTA
    final_dict = {key: seq for key, seq in chain_dict.items() if len(seq) > min_length}

    for chain_id, sequence in final_dict.items():
        fasta_file = os.path.join(fasta_directory, f"{chain_id}.fa")
        with open(fasta_file, "w") as f:
            f.write(f">{chain_id}\n{sequence}\n")
        print(f"FASTA saved: {fasta_file}")

    return final_dict

def get_offset_from_pdb(pdb_id, pdb_directory):
    """Extract offsets per chain from PDB file. Return a dictionary {chain_id: offset}."""

    pdb_file = os.path.join(pdb_directory, f"{pdb_id}.pdb")

    offset_per_chain = {}
    first_missing_per_chain = {}
    first_atom_per_chain = {}

    with open(pdb_file, "r") as f:
        for line in f:
            parts = line.split()
            
            if line.startswith("REMARK 465") and len(parts) == 5:
                _, _, resname, chain, res_id = parts 
                try:
                    chain_id=pdb_id+"_"+chain
                    res_id = int(res_id) 
                    if chain_id not in first_missing_per_chain:
                        first_missing_per_chain[chain_id] = res_id
                except ValueError:
                    pass 

            elif line.startswith("ATOM"):
                try:
                    res_id = int(line[22:26].strip()) 
                    chain = line[21]
                    chain_id=pdb_id+"_"+chain
                    if chain_id not in first_atom_per_chain:
                        first_atom_per_chain[chain_id] = res_id
                except ValueError:
                    pass

    for chain_id in set(first_missing_per_chain) | set(first_atom_per_chain):
        first_res_id = min(
            first_missing_per_chain.get(chain_id, float('inf')),
            first_atom_per_chain.get(chain_id, float('inf'))
        )
        offset_per_chain[chain_id] = first_res_id - 1

    return offset_per_chain

def get_physicochemical_features(pdb_id, pdb_directory, res_dict):
    """Extract physicochemical properties and B-factor for each residue in a PDB file."""
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
        "VAL": [2.50, 1.08, 0.71, 6.00]
        }
    pdb_file = f"{pdb_directory}/{pdb_id}.pdb"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    physicochemical_data = {}

    for chain in model:
        chain_id = chain.id
        for residue in chain:
            if residue.id[0] == " " and "CA" in residue:  # Solo aminoácidos estándar con Cα
                res_id = residue.id[1]  
                res_name = residue.resname  
                b_factor = residue["CA"].bfactor  # B-factor del carbono alfa   
                # Obtener propiedades o usar valores por defecto
                features = AA_PROPERTIES.get(res_name, [0.0] * 7) + [b_factor]
                physicochemical_data[(res_id, chain_id, res_dict[res_name])] = features
    return physicochemical_data

def get_structural_features(pdb_id, pdb_directory, res_dict):
    """Extract structural features from a PDB file and save to a CSV file."""
    pdb_file = f"{pdb_directory}/{pdb_id}.pdb"

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    # Load DSSP (ASA and secondary structure)
    dssp = DSSP(model, pdb_file, dssp="mkdssp") 
    # for key in list(dssp.keys())[:5]:
    #     print(f"Residuo: {key}, Datos DSSP: {dssp[key]}")
    # Compute Half Sphere Exposure
    hse = HSExposureCB(model)
    # Obtener todos los Cα para Contact Number (CN)
    ca_atoms = [residue['CA'] for chain in model for residue in chain if 'CA' in residue]
    ns = NeighborSearch(ca_atoms)

    structural_data={}

    for chain in model:
        for residue in chain:
            chain_id = chain.id
            res_id = residue.id[1]
            res_name = residue.resname  
            if residue.id[0] == " " and (chain_id, res_id) in dssp:
                # Torsional angles phi y psi (sinus and cosinus)
                phi = np.radians(dssp[(chain.id, res_id)][4])
                psi = np.radians(dssp[(chain.id, res_id)][5])
                if phi is None or psi is None:
                    sin_phi, cos_phi, sin_psi, cos_psi = 0.0, 0.0, 0.0, 0.0
                else:
                    phi, psi = np.radians(phi), np.radians(psi)
                    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
                    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
                # Solvent Accessibility ASA
                asa = dssp[(chain.id, res_id)][3]

                # Secondary Structure
                ss=dssp[(chain_id,res_id)][2]

                # Half Sphere Exposure (HSE)
                key = (chain.id, residue.id)  # Esto es ('A', (' ', 1, ' '))
                if key in hse:
                    hse_values = hse[key]
                else:
                    print(f"Residuo {key} no encontrado en HSE")
                    continue  # O maneja el caso de otra forma
                    
                # Contact Number (CN)
                cn = "NA"
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    neighbors = ns.search(ca_atom.coord, 10.0)
                    cn = len(neighbors) - 1

                data_list=[sin_phi, cos_phi, sin_psi, cos_psi,
                    1 if ss == "H" else 0,
                    1 if ss == "B" else 0,
                    1 if ss == "E" else 0,
                    1 if ss == "G" else 0,
                    1 if ss == "I" else 0,
                    1 if ss == "T" else 0,
                    1 if ss == "S" else 0,
                    1 if ss == "-" else 0,                        
                    asa, hse_values[0], hse_values[1], cn]
                structural_data[(res_id,chain_id, res_dict[res_name])]=data_list
    return structural_data

def run_psiblast(chain_dictionary, db, fasta_directory, psiblast_directory, num_iterations=3, evalue=0.001):
    """Run PSI-BLAST on unique sequences and copy results for duplicate chains."""
    
    os.makedirs(psiblast_directory, exist_ok=True)
    # Initialise a dictionary which is reversed from the input dictionary: when executing PSI-BlLAST, key will be sequence and value will be chain_id 
    processed_sequences = {} 

    for chain_id, sequence in chain_dictionary.items():
        query_fasta = f"{fasta_directory}/{chain_id}.fa"
        out_pssm = f"{psiblast_directory}/{chain_id}.pssm"
        
        if sequence in processed_sequences:
            # If PSI-BLAST already done for this sequence, copy PSSM file
            existing_pssm = f"{psiblast_directory}/{processed_sequences[sequence]}.pssm"
            shutil.copy(existing_pssm, out_pssm)
            print(f"Copied PSSM from {processed_sequences[sequence]} to {chain_id}")
        else:
            # Execute PSI-BLAST for new sequence
            cmd = [
                "psiblast", 
                "-query", query_fasta, 
                "-db", db,
                "-num_iterations", str(num_iterations), 
                "-num_threads", "6",
                "-evalue", str(evalue), 
                "-out_ascii_pssm", out_pssm, 
                "-outfmt", "0"
            ]
            with open(os.devnull, 'w') as devnull:
                try:
                    subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
                    print(f"PSI-BLAST completed for {chain_id}. PSSM saved to {out_pssm}")
                    # Add proccessed chain_id to the dictionary with its sequence as the key
                    processed_sequences[sequence] = chain_id  
                except subprocess.CalledProcessError as e:
                    print(f"Error running PSI-BLAST for {chain_id}: {e}")

def run_jackhmmer(chain_dictionary, db, fasta_directory, jackhmmr_directory, num_iterations=5):
    """Run jackhmmer and save the final model in jackhmmr directory."""

    os.makedirs(jackhmmr_directory, exist_ok=True)

    # Initialise a dictionary which is reversed from the input dictionary: when executing PSI-BlLAST, key will be sequence and value will be chain_id 
    processed_sequences = {} 
    
    for chain_id, sequence in chain_dictionary.items():
        query_fasta = f"{fasta_directory}/{chain_id}.fa"
        out_hmm = f"{jackhmmr_directory}/{chain_id}.hmm"
        
        if sequence in processed_sequences:
            # If jackhmmr already done for this sequence, copy PSSM file
            existing_hmm = f"{jackhmmr_directory}/{processed_sequences[sequence]}.hmm"
            shutil.copy(existing_hmm, out_hmm)
            print(f"Copied HMM from {processed_sequences[sequence]} to {chain_id}")
        else:
            # Execute jackhmmr for new sequence
            cmd = [
                "jackhmmer", 
                "-N", str(num_iterations), 
                "--chkhmm", f"{jackhmmr_directory}/model", 
                "--cpu", "6", 
                query_fasta, 
                db
            ]
            with open(os.devnull, 'w') as devnull:
                try:
                    subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
                    # Search for latest model
                    hmm_files = glob.glob(f"{jackhmmr_directory}/model-*.hmm")
                    if hmm_files:
                        latest_model = max(hmm_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
                        final_hmm_path = f"{jackhmmr_directory}/{chain_id}.hmm"
                        os.rename(latest_model, final_hmm_path)
                        
                        # Delete temp files model-*.hmm
                        for hmm_file in hmm_files:
                            try:
                                os.remove(hmm_file)
                            except FileNotFoundError:
                                pass
                        print(f"jackhmmer completed successfully. Model saved to {final_hmm_path}")
                        processed_sequences[sequence] = chain_id  
                    else:
                        print("No HMM model files found.")
                except subprocess.CalledProcessError as e:
                    print(f"Error running jackhmmer: {e}")

def parse_pssm(chain_dictionary, psiblast_directory, offset_per_chain):
    """Parse the PSSM file and return a dictionary with residue indices as keys and PSSM values as lists."""
    pssm_data = {}
    for chain_id, _ in chain_dictionary.items():
        pssm_file=f"{psiblast_directory}/{chain_id}.pssm"
        start = False
        with open(pssm_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith("A"):
                    start = True
                    continue
                if start:
                    scoreList=line.split()
                    if len(scoreList) < 20:
                        break 
                    offset=offset_per_chain[chain_id]
                    res_id=int(scoreList[0])+offset
                    res_name = scoreList[1]
                    scores = list(map(int, scoreList[0:22][2:]))
                    sigmoid_scores=[sigmoid(s) for s in scores]
                    pssm_data[(res_id, chain_id.split("_")[1], res_name)] = sigmoid_scores
    return pssm_data
    
def parse_hmm(chain_dictionary, jackhmmr_directory, offset_per_chain):
    """Parse the HMM file and return a dictionary with residue indices as keys and HMM probabilities."""
    hmm_data = {}
    for chain_id, sequence in chain_dictionary.items():
        hmm_file=f"{jackhmmr_directory}/{chain_id}.hmm"
        reading = False
        emissions = None
        with open(hmm_file) as f:
            for line in f:
                if line.startswith("HMM "):
                    reading = True
                    continue
                if reading:
                    scores = line.split()
                    if len(scores)==26:
                        hmm_index = int(scores[0]) - 1 
                        offset=offset_per_chain[chain_id]
                        res_id=hmm_index + 1 + offset
                        res_name=sequence[hmm_index]
                        emissions = [float(s) for s in scores[1:21]]
                    if len(scores)==7 and not scores[0].startswith("m->"):
                        transitions = [0.0 if s == '*' else float(s) for s in scores]
                        if emissions and transitions:
                            hmm_data[(res_id, chain_id.split("_")[1], res_name)]=emissions+transitions
                            emissions=None
    return hmm_data

column_names = (
    ["prop1","prop2","prop3","prop4", "b_factor"]
    + ["sin_phi", "cos_phi", "sin_psi", "cos_psi", "H", "B", "E", "G", "I", "T", "S", "-", "ASA", "HSE_up", "HSE_down", "CN"]
    + ["PSSM_" + str(i) for i in range(1,21)]
    + ["HMM_" + str(i) for i in range(1,28)]
    + ["binding_site"]
)

def get_binding_sites(pdb_id, pdb_site_directory, res_dict):
    """Parse edited PDB files and return a dictionary labeling binding site residues."""

    pdb_file = os.path.join(pdb_site_directory, f"{pdb_id}_protwithsite.pdb")
    binding_data = {}

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("SITE"):  
                try:
                    res_id = int(line[22:26].strip())
                    chain = line[21]
                    res_name = line[17:20].strip() 
                    if line.startswith("SITE"):
                        binding_data[(res_id, chain, res_dict[res_name])] = [1]
                    elif (res_id, chain) not in binding_data:
                        binding_data[(res_id, chain, res_dict[res_name])] = [0]  

                except ValueError:
                    pass
    return binding_data

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
        print("--------------No hay datos para escribir en el CSV.")
        return
    
    if mismatch_found:
        print("--------------Se encontraron residuos que no coinciden. Ver errores.log")

    # Convert to DataFrame
    residue_ids = np.array(list(final_data.keys()), dtype=object)
    features = np.array(list(final_data.values()))
    df = pd.DataFrame(features, index=residue_ids, columns=column_names)
    # Save as CSV
    df.to_csv(output_file, index_label="Residue_ID")
    print(f"--------------Archivo guardado: {output_file}")



def main():

    with open("errores.log", "w") as f:
        f.write("")

    res_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

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

    # pdb_filtered = ["1a08"]
    for pdb_id in pdb_filtered[0:2001]:
        try:
            # Start timer
            start_time = time.time() 

            # Extract fasta for each chain
            chains=pdb2fasta(pdb_id,  pdb_directory, fasta_directory, res_dict)
            offset=get_offset_from_pdb(pdb_id, pdb_directory)
            # Get physicochemical and structural information from the whole pdb
            physicochemical_data=get_physicochemical_features(pdb_id, pdb_directory, res_dict)
            structural_data=get_structural_features(pdb_id, pdb_directory, res_dict)
            # Run PSI-BLAST for each chain
            run_psiblast(chains, database_psiblast, fasta_directory, psiblast_directory)
            pssm_data=parse_pssm(chains, psiblast_directory, offset)
            # Run jackhmmr for each chain
            run_jackhmmer(chains, database_jackhmmer, fasta_directory, jackhmmr_directory,num_iterations=3)
            hmm_data=parse_hmm(chains, jackhmmr_directory, offset)
            # Get binding site label
            binding_data=get_binding_sites(pdb_id, pdb_site_directory, res_dict)
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
            with open("errores.log", "a") as f:
                f.write(f"{pdb_id}: {e}\n")
                f.write(traceback.format_exc() + "\n")

if __name__ == "__main__":
    main()
