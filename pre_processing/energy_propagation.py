import heapq
import numpy as np
from Bio import PDB
import requests
from pdb_preprocessing import ProteinStructureProcessor

class EnergyPropagation:
    def __init__(self, protein_structure_processor, allosteric_site, bond_strengths):
        self.processor = protein_structure_processor
        self.allosteric_site = allosteric_site
        self.bond_strengths = bond_strengths  # A dict or matrix of bond strengths between residues
        self.energy_propagation = {}  # Store energy propagation values for each residue

    def calculate_propagation_energy(self, bond_strength, distance):
        """Calculate energy propagation based on bond strength and distance."""
        return bond_strength / distance if distance > 0 else 0
    
    def propagate_energy(self):
        """Perform BFS to propagate energy from the allosteric site through the graph."""
        num_residues = len(self.processor.residues)
        distances = self.processor.distance_matrix
        bond_strengths = self.bond_strengths

        # Initialize min-heap priority queue for BFS and a dictionary to store visited residues
        pq = []
        heapq.heappush(pq, (0, self.allosteric_site))  # (initial_energy, residue_index)
        visited = [False] * num_residues
        
        # Initialize energy propagation dictionary with "infinity" for all residues except the source
        self.energy_propagation = {i: float('inf') for i in range(num_residues)}
        self.energy_propagation[self.allosteric_site] = 0
        
        while pq:
            current_energy, current_residue = heapq.heappop(pq)
            
            if visited[current_residue]:
                continue
            
            visited[current_residue] = True
            
            # Propagate to all interacting residues
            for neighbor in range(num_residues):
                if self.processor.adjacency_matrix[current_residue][neighbor] == 1:
                    # Calculate the propagation energy to the neighboring residue
                    bond_strength = self.bond_strengths.get((current_residue, neighbor), 1)
                    distance = distances[current_residue][neighbor]
                    energy_to_propagate = self.calculate_propagation_energy(bond_strength, distance)
                    
                    # Update the energy propagation if we find a lower energy path
                    if current_energy + energy_to_propagate < self.energy_propagation[neighbor]:
                        self.energy_propagation[neighbor] = current_energy + energy_to_propagate
                        heapq.heappush(pq, (self.energy_propagation[neighbor], neighbor))
    
    def get_energy_propagation(self):
        """Return the energy propagation values for all residues."""
        return self.energy_propagation

    def calculate_energy_propagation_for_residue(self, residue_index):
        """Calculate energy propagation from a given residue to its neighbors."""
        energy_map = {}
        for neighbor in range(len(self.processor.residues)):
            if self.processor.adjacency_matrix[residue_index][neighbor] == 1:
                bond_strength = self.bond_strengths.get((residue_index, neighbor), 1)
                distance = self.processor.distance_matrix[residue_index][neighbor]
                energy_map[neighbor] = self.calculate_propagation_energy(bond_strength, distance)
        
        return energy_map

# Example usage
if __name__ == "__main__":
    # Load protein structure
    pdb_id = "4WB7"
    allosteric_site = 0  # Index of the residue acting as the allosteric site

    # Initialize the processor and energy propagation classes
    protein_processor = ProteinStructureProcessor(pdb_id)
    energy_propagation = EnergyPropagation(protein_processor, allosteric_site, protein_processor.bond_strengths)

    # Perform energy propagation calculations
    energy_propagation.propagate_energy()

    # Retrieve energy propagation results
    propagation_results = energy_propagation.get_energy_propagation()
    print("Energy Propagation Results:", propagation_results)
