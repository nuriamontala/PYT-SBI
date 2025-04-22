from Bio.PDB import PDBParser, DSSP, HSExposureCB, NeighborSearch
import numpy as np

def get_structural_features(pdb_id, pdb_directory):
    """Extract structural features from a PDB file and save to a CSV file."""

    res_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    
    pdb_file = f"{pdb_directory}/{pdb_id}.pdb"

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    # Load DSSP (ASA and secondary structure)
    dssp = DSSP(model, pdb_file, dssp="mkdssp") 
 
    # Compute Half Sphere Exposure
    hse = HSExposureCB(model)

    # Get all Cα for Contact Number (CN)
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
                    print(f"Residue {key} not found in HSE")
                    continue  
                    
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
