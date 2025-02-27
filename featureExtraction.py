import os
import subprocess
from Bio.PDB import PDBParser, DSSP, HSExposureCB, NeighborSearch
import glob
import pandas as pd
import numpy as np
import shutil
import time

AA_PROPERTIES = {
    "ALA": [1.24, 0.62, 0.38, 0.05, 6.00, 1.42, 1.00],
    "ARG": [2.74, -2.53, 0.89, 0.93, 10.76, 0.98, 0.79],
    "ASN": [2.14, -0.78, 0.48, 0.54, 5.41, 0.89, 0.76],
    "ASP": [2.16, -0.90, 0.49, 0.52, 2.85, 0.85, 0.72],
    "CYS": [1.50, 0.29, 0.54, 0.79, 5.07, 0.77, 0.63],
    "GLN": [2.17, -0.85, 0.51, 0.70, 5.65, 1.11, 0.92],
    "GLU": [2.18, -0.74, 0.52, 0.67, 3.15, 1.09, 0.90],
    "GLY": [0.00, 0.48, 0.00, 0.00, 6.06, 0.57, 0.57],
    "HIS": [2.48, -0.40, 0.69, 0.95, 7.60, 1.00, 0.87],
    "ILE": [3.08, 1.25, 0.76, 0.90, 6.05, 1.00, 0.97],
    "LEU": [2.80, 1.22, 0.74, 0.92, 6.01, 1.34, 1.01],
    "LYS": [2.90, -2.25, 0.84, 0.93, 9.74, 1.00, 0.79],
    "MET": [2.67, 1.02, 0.72, 1.01, 5.74, 1.20, 0.97],
    "PHE": [2.58, 1.47, 0.78, 1.26, 5.48, 1.16, 0.95],
    "PRO": [1.95, 0.09, 0.64, 0.75, 6.30, 0.57, 0.55],
    "SER": [1.31, -0.28, 0.41, 0.68, 5.68, 0.77, 0.75],
    "THR": [1.50, -0.18, 0.44, 0.69, 5.60, 0.83, 0.78],
    "TRP": [3.07, 1.45, 0.81, 1.41, 5.89, 1.09, 0.92],
    "TYR": [2.67, 0.94, 0.76, 1.20, 5.64, 0.97, 0.90],
    "VAL": [2.50, 1.08, 0.71, 0.91, 6.00, 1.06, 0.93],
}
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pdb2fasta(pdb_id, pdb_directory, fasta_directory):
    """Extract protein sequence from a PDB file and save it as FASTA in the fasta directory."""
    res_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    
    os.makedirs(pdb_directory, exist_ok=True)
    os.makedirs(fasta_directory, exist_ok=True)

    pdb_file = f"{pdb_directory}/{pdb_id}.pdb"
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    
    chain_dict={}
    for model in structure:
        for chain in model:
            chain_id=pdb_id+"_"+chain.get_id()
            seq=[]
            for residue in chain:
                if residue.resname in res_dict:
                    seq.append(res_dict[residue.resname])
            seq=''.join(seq)
            if len(seq)>30:
                chain_dict[chain_id]=seq
            seq=[]
    
    for chain_id in chain_dict:
        fasta_file = f"{fasta_directory}/{chain_id}.fa"
        with open(fasta_file, "w") as f:
            f.write(f'>{chain_id}\n'+''.join(chain_dict[chain_id]))
            print(f"Fasta sequence saved to {fasta_file}")
    return chain_dict

def get_physicochemical_features(pdb_id, pdb_directory):
    """Extract physicochemical properties and B-factor for each residue in a PDB file."""
    
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
                aa = residue.resname  
                b_factor = residue["CA"].bfactor  # B-factor del carbono alfa                
                # Obtener propiedades o usar valores por defecto
                features = AA_PROPERTIES.get(aa, [0.0] * 7) + [b_factor]
                physicochemical_data[(res_id, chain_id)] = features

    return physicochemical_data

def get_structural_features(pdb_id, pdb_directory, output_directory):
    """Extract structural features from a PDB file and save to a CSV file."""
    pdb_file = f"{pdb_directory}/{pdb_id}.pdb"

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    # Load DSSP (ASA and secondary structure)
    dssp = DSSP(model, pdb_file, dssp="mkdssp")  

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
                structural_data[(res_id,chain_id)]=data_list
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

def parse_pssm(chain_dictionary, psiblast_directory):
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
                    res_id=int(scoreList[0])
                    scores = list(map(int, scoreList[0:22][2:]))
                    sigmoid_scores=[sigmoid(s) for s in scores]
                    pssm_data[(res_id, chain_id.split("_")[1])] = sigmoid_scores

    return(pssm_data)
    
def parse_hmm(chain_dictionary, jackhmmr_directory):
    """Parse the HMM file and return a dictionary with residue indices as keys and HMM probabilities."""
    hmm_data = {}
    for chain_id, _ in chain_dictionary.items():
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
                        res_id=scores[0]
                        emissions = [float(s) / 10000 for s in scores[1:21]]
                    if len(scores)==7 and not scores[0].startswith("m->"):
                        transitions = [0.0 if s == '*' else float(s) / 10000 for s in scores]
                        if emissions and transitions:
                            hmm_data[(int(res_id), chain_id.split("_")[1])]=emissions+transitions
                            emissions=None

    return hmm_data

column_names = (
    ["prop1","prop2","prop3","prop4","prop5","prop6","prop7", "b_factor"]
    + ["sin_phi", "cos_phi", "sin_psi", "cos_psi", "H", "B", "E", "G", "I", "T", "S", "-", "ASA", "HSE_up", "HSE_down", "CN"]
    + ["PSSM_" + str(i) for i in range(1,21)]
    + ["HMM_" + str(i) for i in range(1,28)]
)

def merge_data(physicochemical_data, structural_data, pssm_data, hmm_data):
    final_data={}
    for key in structural_data.keys():
        if key in pssm_data.keys() and key in hmm_data.keys():
            final_data[key] = physicochemical_data[key] + structural_data[key] + pssm_data[key] + hmm_data[key]
    
    residue_ids = np.array(list(final_data.keys()), dtype=object)  
    features = np.array(list(final_data.values()))
    return residue_ids, features


def merge_data_to_csv(physicochemical_data, structural_data, pssm_data, hmm_data, output_file):
    final_data = {}

    for key in structural_data.keys():
        if key in pssm_data and key in hmm_data:  
            final_data[key] = physicochemical_data[key] + structural_data[key] + pssm_data[key] + hmm_data[key]

    if not final_data:  # Si no hay datos, no se escribe el CSV
        print("--------------No hay datos para escribir en el CSV.")
        return

    # Convertir a DataFrame
    residue_ids = np.array(list(final_data.keys()), dtype=object)
    features = np.array(list(final_data.values()))

    df = pd.DataFrame(features, index=residue_ids, columns=column_names)

    # Guardar en CSV
    df.to_csv(output_file, index_label="Residue_ID")
    print(f"--------------Archivo guardado: {output_file}")



def main():

    pdb_directory="../files/pdb/original_files"
    fasta_directory="../files/fasta"
    psiblast_directory="../files/psiblast"
    jackhmmr_directory="../files/jackhmmr"
    structural_directory = "../files/structural"
    features_directory = "../files/features"
    database_psiblast = "../database/uniprot_sprot_db"
    database_jackhmmer = "../database/uniprot_sprot.fasta"

    with open("log") as f:
        pdb_missing = {line.strip() for line in f}

    pdb_ids = [f[:-4] for f in os.listdir(pdb_directory) if f.endswith(".pdb")]
    pdb_filtered = [pdb_id for pdb_id in pdb_ids if pdb_id not in pdb_missing]
    for pdb_id in pdb_filtered[5:10]:
        # Start timer
        start_time = time.time() 

        # Extract fasta for each chain
        chains=pdb2fasta(pdb_id,  pdb_directory, fasta_directory)
        # Get physicochemical and structural information from the whole pdb
        physicochemical_data=get_physicochemical_features(pdb_id, pdb_directory)
        structural_data=get_structural_features(pdb_id, pdb_directory, structural_directory)
        # Run PSI-BLAST for each chain
        run_psiblast(chains, database_psiblast, fasta_directory, psiblast_directory)
        pssm_data=parse_pssm(chains, psiblast_directory)
        # Run jackhmmr for each chain
        run_jackhmmer(chains, database_jackhmmer, fasta_directory, jackhmmr_directory,num_iterations=3)
        hmm_data=parse_hmm(chains, jackhmmr_directory)
        # Merge all the features in one numpy matrix
        # residue_ids, features = merge_data(physicochemical_data, structural_data, pssm_data, hmm_data)
        merge_data_to_csv(physicochemical_data, structural_data, pssm_data, hmm_data, f"{features_directory}/{pdb_id}.csv")
        # num_residues, num_features = features.shape
        # print(f"Matrix size: {num_residues} x {num_features}")
        # Save matrix to features directory
        # os.makedirs(features_directory, exist_ok=True)
        # np.savez(f"{features_directory}/{pdb_id}.npz", res_ids=residue_ids, features=features)

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

if __name__ == "__main__":
    main()
