import os
import subprocess
import glob
import shutil
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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