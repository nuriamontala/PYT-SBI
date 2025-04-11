import os
import pandas as pd
import numpy as np
from sequenceProfiles import *
from parsePDB import *
from physchemFeatures import *
from structuralFeatures import *

def merge_data_to_csv(physicochemical_data, structural_data, pssm_data, hmm_data, binding_data, features_directory, pdb_id):
    """Merge all feature dictionaries into a single DataFrame and save it as a CSV."""
    output_file=f"{features_directory}/{pdb_id}.csv"
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

def save_coordinates_to_csv(coordinates_dict, coordinates_directory, pdb_id):
    """Save 3D coordinates of residues to a CSV file."""
    os.makedirs(coordinates_directory, exist_ok=True)
    output_file=f"{coordinates_directory}/{pdb_id}_coord.csv"

    if not coordinates_dict:
        print("--------------No coordinates to write in the CSV.")
        return

    # Extract IDs and coordinates
    residue_ids = np.array(list(coordinates_dict.keys()), dtype=object)
    coords = np.array(list(coordinates_dict.values()))
    
    # Create DataFrame
    df = pd.DataFrame(coords, index=residue_ids, columns=["x", "y", "z"])
    
    # Saves as CSV
    df.to_csv(output_file, index_label="Residue_ID")
    print(f"--------------Coordinates saved: {output_file}")

class PDBObject:
    """
    PDBObject represents a protein complex and encapsulates all processing steps 
    required to extract residue-level features and coordinates.

    Attributes
    ----------
    pdb_id : str
        Identifier of the PDB complex.
    chains : dict
        Dictionary mapping chain IDs to sequences.
    offset : dict
        Dictionary with residue offset information.
    physicochemical : dict
        Computed physicochemical features.
    structural : dict
        Computed structural features.
    pssm : dict
        Parsed PSSM profiles.
    hmm : dict
        Parsed HMM profiles.
    binding_sites : dict
        Binding site labels.
    coordinates : dict
        3D coordinates of residues.
    """
    def __init__(self, pdb_id, dirs, dbs):
        """
        Initialize the PDBObject.

        Parameters
        ----------
        pdb_id : str
            Identifier of the PDB complex.
        dirs : dict
            Dictionary of directory paths for I/O operations.
        dbs : dict
            Dictionary with paths to evolutionary search databases.
        """
        self.pdb_id = pdb_id
        self._dirs = dirs
        self._dbs = dbs

        self.chains = pdb2fasta(self.pdb_id, self._dirs["pdb"], self._dirs["fasta"])
        self.offset = get_offset_from_pdb(self.pdb_id, self._dirs["pdb"])
        self.physicochemical = {}
        self.structural = {}
        self.pssm = {}
        self.hmm = {}
        self.binding_sites = {}
        self.coordinates = {}

    def compute_features(self):
        """
        Run all feature extraction methods and store the results as attributes.

        Includes physicochemical, structural, PSSM, HMM, binding site, and
        3D coordinate features.
        """
        self.physicochemical = get_physicochemical_features(self.pdb_id, self._dirs["pdb"])
        self.structural = get_structural_features(self.pdb_id, self._dirs["pdb"])

        run_psiblast(self.chains, self._dbs["psiblast"], self._dirs["fasta"], self._dirs["psiblast"])
        self.pssm = parse_pssm(self.chains, self._dirs["psiblast"], self.offset)

        run_jackhmmer(self.chains, self._dbs["jackhmmer"], self._dirs["fasta"], self._dirs["jackhmmer"], num_iterations=3)
        self.hmm = parse_hmm(self.chains, self._dirs["jackhmmer"], self.offset)

        self.binding_sites = get_binding_sites(self.pdb_id, self._dirs["sites"])
        self.coordinates = get_coordinates(self.pdb_id, self._dirs["pdb"])

    def save_to_csv(self):
        """Save all extracted features and coordinates to CSV files in the corresponding directories."""
        merge_data_to_csv(
            self.physicochemical,
            self.structural,
            self.pssm,
            self.hmm,
            self.binding_sites,
            self._dirs["features"],
            self.pdb_id
        )
        save_coordinates_to_csv(self.coordinates, self._dirs["coords"], self.pdb_id)
    
    def clean_tmp(self):
        """Delete all intermediate files generated in temporary folders."""
        for folder in [self._dirs["fasta"], self._dirs["psiblast"], self._dirs["jackhmmer"]]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    os.remove(os.path.join(folder, file))    

    def run_all(self):
        """Run compute_features, save_to_csv and clean_tmp at the same time."""
        self.compute_features()
        self.save_to_csv()
        self.clean_tmp()