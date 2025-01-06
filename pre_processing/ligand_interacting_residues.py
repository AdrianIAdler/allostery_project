from Bio import PDB
import numpy as np
import heapq
from Bio import PDB
import requests

class LigandInteractionFinder:
    def __init__(self, pdb_id, ligand_chain, ligand_residue_id):
        self.pdb_id = pdb_id
        self.ligand_chain = ligand_chain
        self.ligand_residue_id = ligand_residue_id
        self.structure = self.load_structure()

    def load_structure(self):
        """Load the PDB structure from RCSB and return the structure object."""
        pdb_parser = PDB.PDBParser(QUIET=True)
        pdb_filename = f"{self.pdb_id}.pdb"
        
        # Download the PDB file if it doesn't already exist
        url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
        response = requests.get(url)
        with open(pdb_filename, "w") as pdb_file:
            pdb_file.write(response.text)

        structure = pdb_parser.get_structure(self.pdb_id, pdb_filename)
        return structure

    def find_ligand_residues(self):
        """Find residues in proximity to the ligand."""
        # Get the ligand atoms
        ligand_atoms = []
        ligand_found = False
        for model in self.structure:
            for chain in model:
                if chain.id == self.ligand_chain:
                    for residue in chain:
                        if residue.id == self.ligand_residue_id:
                            ligand_atoms.extend(residue.get_atoms())
                            ligand_found = True
                            print(f"Ligand found: {residue.get_resname()} {residue.id} in chain {chain.id}")
        
        if not ligand_found:
            print(f"Ligand with chain '{self.ligand_chain}' and residue ID '{self.ligand_residue_id}' not found.")
            return

        # Find interacting residues based on distance cutoff
        interacting_residues = set()
        distance_cutoff = 5.0  # Angstroms

        # Iterate over residues in the structure to find those close to the ligand
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.id != self.ligand_residue_id and PDB.is_aa(residue):
                        for atom in residue:
                            for ligand_atom in ligand_atoms:
                                distance = ligand_atom - atom  # Calculate Euclidean distance
                                if distance <= distance_cutoff:
                                    interacting_residues.add(residue)
                                    break  # No need to check further if a contact is found

        if not interacting_residues:
            print(f"No interacting residues found within {distance_cutoff} Å.")
        else:
            # Print interacting residues
            for residue in interacting_residues:
                print(f"Interacting Residue: {residue.get_resname()} {residue.id} in chain {residue.parent.id}")

# Usage Example
ligand_finder = LigandInteractionFinder('4WB7', 'A', (' ', 401, ' '))
ligand_finder.find_ligand_residues()

#how to get the ligand list from pdb --> on interface have a list of ligands
