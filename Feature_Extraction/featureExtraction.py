import time
import traceback
from pdbObject import PDBObject 

def extract_features():
    # Clean previous logs
    with open("errores.log", "w") as f:
        f.write("")
    with open("errors.log", "w") as f:
        f.write("")

    # Define paths y and databases
    dirs = {
        "pdb": "../files/pdb/original_files",
        "sites": "../files/pdb/protwithsitefinal",
        "fasta": "../files/fasta",
        "psiblast": "../files/psiblast",
        "jackhmmer": "../files/jackhmmer",
        "features": "../files/features",
        "coords": "../files/coordinates"
    }

    dbs = {
        "psiblast": "../database/uniprot_sprot_db",
        "jackhmmer": "../database/uniprot_sprot.fasta"
    }

    # List of PDB to process
    with open("all_pdbs.txt") as f:
        pdb_list = [line.strip() for line in f if line.strip()]

    # Iterate through each PDB
    for pdb_id in pdb_list[:1]:
        print(f"============== Processing {pdb_id}.pdb ==============")
        start_time = time.time()
        try:
            pdb_obj = PDBObject(pdb_id, dirs, dbs)
            pdb_obj.run_all()
        except Exception as e:
            with open("errors.log", "a") as f:
                f.write(f"{pdb_id}: {e}\n")
                f.write(traceback.format_exc() + "\n")
            print(f"--------------ERROR processing {pdb_id}, check errors.log.")
        else:
            elapsed = time.time() - start_time
            print(f"--------------Finished {pdb_id} in {elapsed:.2f} seconds")

if __name__ == "__main__":
    extract_features()
