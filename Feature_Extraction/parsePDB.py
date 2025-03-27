import os

def pdb2fasta(pdb_id, pdb_directory, fasta_directory, min_length=30, max_chains=6):
    """Extract protein sequence from a PDB file and save it as FASTA in the fasta directory."""

    res_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
       
    os.makedirs(fasta_directory, exist_ok=True)

    pdb_file = os.path.join(pdb_directory, f"{pdb_id}.pdb")
    
    # Dictionary to store all sequences
    chain_dict = {}

    # Read sequence from SEQRES section in the PDB
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("SEQRES"):
                parts = line.split()
                chain_id=pdb_id+"_"+parts[2]
                # Convert 3-letter code to 1-letter
                seq = [res_dict.get(res, "X") for res in parts[4:]]  
                if chain_id in chain_dict:
                    chain_dict[chain_id] += "".join(seq)
                else:
                    chain_dict[chain_id] = "".join(seq)

    num_chains = len(chain_dict)
    if num_chains > max_chains:
        raise Exception(f"Skipping PDB. File has {num_chains} chains (limit: {max_chains})")
    # Filter short chains and save in FASTA
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

def get_binding_sites(pdb_id, pdb_site_directory):
    """Parse edited PDB files and return a dictionary labeling binding site residues."""

    res_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    
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