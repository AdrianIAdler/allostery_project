import numpy as np
import heapq
from Bio import PDB
import requests

class ProteinStructureProcessor:
    def __init__(self, pdb_id, chain_id=None):
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.pdb_file = self.get_pdb_file(pdb_id)
        
        if chain_id:
            self.pdb_file = self.extract_chain(chain_id)  # Update to chain-specific file
        
        self.residues = self.load_protein_structure(self.pdb_file)
        self.adjacency_matrix = self.create_adjacency_matrix()
        self.distance_matrix = self.create_distance_matrix()
        self.bond_strengths = self.estimate_bond_strengths()

    def get_pdb_file(self, pdb_id):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        filename = f"{pdb_id}.pdb"
        with open(filename, "w") as pdb_file:
            pdb_file.write(response.text)
        return filename

    def extract_chain(self, chain_id):
        """
        Extracts a specific chain from the PDB file and writes it to a new file.

        Args:
            chain_id: The chain ID to extract (e.g., "A").
        
        Returns:
            str: The filename of the extracted chain PDB.
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb_file)
        output_file = f"{self.pdb_id}_{chain_id}.pdb"

        # Extract the chain
        model = structure[0]  # Use the first model
        chain = model[chain_id]  # Get the specified chain

        # Save the chain to a new PDB file
        io = PDB.PDBIO()
        io.set_structure(chain)
        io.save(output_file)
        print(f"Chain {chain_id} saved to {output_file}")
        return output_file

    def load_protein_structure(self, pdb_file):
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('Protein', pdb_file)
        residues = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.Polypeptide.is_aa(residue):
                        coord = residue['CA'].get_coord()
                        residues.append({
                            'name': residue.get_resname(),
                            'id': residue.get_id(),
                            'coordinates': coord
                        })
        return residues

    def create_adjacency_matrix(self):
        num_residues = len(self.residues)
        adjacency_matrix = np.zeros((num_residues, num_residues), dtype=int)
        cutoff_distance = 8.0
        
        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                distance = self.calculate_distance(self.residues[i], self.residues[j])
                if distance <= cutoff_distance:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
        return adjacency_matrix

    def create_distance_matrix(self):
        num_residues = len(self.residues)
        distance_matrix = np.zeros((num_residues, num_residues))

        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                distance_matrix[i][j] = self.calculate_distance(self.residues[i], self.residues[j])
                distance_matrix[j][i] = distance_matrix[i][j]  # Mirror value for symmetric matrix
        return distance_matrix

    def calculate_distance(self, residue_a, residue_b):
        coord_a = residue_a['coordinates']
        coord_b = residue_b['coordinates']
        return np.linalg.norm(coord_a - coord_b)
    
    def estimate_bond_strengths(self):
        bond_strengths = {}
        for i in range(len(self.residues)):
            for j in range(i + 1, len(self.residues)):
                if self.adjacency_matrix[i][j] == 1:
                    distance = self.distance_matrix[i][j]
                    bond_strength = self.calculate_bond_strength(distance, self.residues[i], self.residues[j])
                    bond_strengths[(i, j)] = bond_strength
                    bond_strengths[(j, i)] = bond_strength  # Since it’s a symmetric bond
        return bond_strengths

    def calculate_bond_strength(self, distance, residue_a, residue_b):
        if distance < 4.0:
            return 10.0  # Strong bond, possibly covalent
        elif distance < 6.0:
            return 5.0  # Moderate bond, possibly hydrogen bond
        else:
            return 1.0  # Weak bond, possibly van der Waals force

    def get_residue_positions(self):
        positions = []
        for residue in self.residues:
            atom_coords = [atom.get_coord() for atom in residue.get_atoms()]
            centroid = np.mean(atom_coords, axis=0)
            positions.append(centroid)
        return positions


# Process the full protein
processor_full = ProteinStructureProcessor('4WB7')

# Process only chain A
processor_chain_a = ProteinStructureProcessor('4WB7', chain_id='A')

# Access results for the chain
print("Residues in Chain A:", processor_chain_a.residues)
print("Adjacency Matrix for Chain A:\n", processor_chain_a.adjacency_matrix)
print("Distance Matrix for Chain A:\n", processor_chain_a.distance_matrix)


## line 120 residues has S, missing "for residue in residues"`? 