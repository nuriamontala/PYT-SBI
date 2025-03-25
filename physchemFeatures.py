from Bio.PDB import PDBParser

def get_physicochemical_features(pdb_id, pdb_directory):
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
    
    res_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    
    pdb_file = f"{pdb_directory}/{pdb_id}.pdb"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    physicochemical_data = {}

    for chain in model:
        chain_id = chain.id
        for residue in chain:
             # Only standard aminoacids with Cα
            if residue.id[0] == " " and "CA" in residue: 
                res_id = residue.id[1]  
                res_name = residue.resname  
                b_factor = residue["CA"].bfactor  
                features = AA_PROPERTIES.get(res_name, [0.0] * 7) + [b_factor]
                physicochemical_data[(res_id, chain_id, res_dict[res_name])] = features
    return physicochemical_data