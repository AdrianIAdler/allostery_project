import numpy as np
import heapq
from Bio import PDB
import requests
from pdb_preprocessing import ProteinStructureProcessor
from energy_propagation import EnergyPropagation


class CommunicationPathways:
    def __init__(self, protein_structure_processor, energy_propagation):
        self.processor = protein_structure_processor
        self.energy_propagation = energy_propagation
        self.shortest_paths = {}  # Store shortest paths from a residue to all other residues

    def calculate_shortest_path(self, start_residue):
        """Calculate the shortest path from start_residue to all other residues using Dijkstra's algorithm."""
        num_residues = len(self.processor.residues)
        distances = {i: float('inf') for i in range(num_residues)}
        distances[start_residue] = 0
        pq = [(0, start_residue)]  # Priority queue with (distance, residue)
        
        while pq:
            current_distance, current_residue = heapq.heappop(pq)
            
            if current_distance > distances[current_residue]:
                continue

            for neighbor in range(num_residues):
                if self.processor.adjacency_matrix[current_residue][neighbor] == 1:
                    # Calculate distance as energy cost for this pathway
                    bond_strength = self.energy_propagation.bond_strengths.get((current_residue, neighbor), 1)
                    distance = self.processor.distance_matrix[current_residue][neighbor]
                    edge_weight = self.energy_propagation.calculate_propagation_energy(bond_strength, distance)
                    
                    new_distance = current_distance + edge_weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        heapq.heappush(pq, (new_distance, neighbor))

        self.shortest_paths[start_residue] = distances

    def get_shortest_path(self, start_residue, end_residue):
        """Return the shortest path distance from start_residue to end_residue."""
        if start_residue not in self.shortest_paths:
            self.calculate_shortest_path(start_residue)
        return self.shortest_paths[start_residue].get(end_residue, float('inf'))

    def get_all_paths_from_allosteric_site(self, allosteric_site):
        """Calculate shortest paths from the allosteric site to all residues."""
        self.calculate_shortest_path(allosteric_site)
        return self.shortest_paths[allosteric_site]

# Example usage
if __name__ == "__main__":
    # Load protein structure and define bond strengths
    pdb_id = "4WB7"  # Replace with your PDB file
    allosteric_site = 0  # Index of the residue acting as the allosteric site
    active_site = 398      # Example active site residue index

    # Initialize the processor and energy propagation classes
    protein_processor = ProteinStructureProcessor(pdb_id, chain_id= "A")
    energy_propagation = EnergyPropagation(protein_processor, allosteric_site, protein_processor.bond_strengths)
    energy_propagation.propagate_energy()

    # Initialize communication pathway analysis
    pathways = CommunicationPathways(protein_processor, energy_propagation)

    # Get the shortest communication pathway from the allosteric site to the active site
    shortest_path_distance = pathways.get_shortest_path(allosteric_site, active_site)
    print(f"Shortest path from allosteric site {allosteric_site} to active site {active_site}: {shortest_path_distance}")

    # Get shortest paths from allosteric site to all residues
    all_paths_from_allosteric_site = pathways.get_all_paths_from_allosteric_site(allosteric_site)
    print("Shortest paths from allosteric site to all residues:", all_paths_from_allosteric_site)
