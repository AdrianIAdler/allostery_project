import numpy as np
import heapq
from Bio import PDB
import requests
from pdb_preprocessing import ProteinStructureProcessor
from energy_propagation import EnergyPropagation
from structural_perturbation import StructuralPerturbation

class CorrelationAnalysis:
    def __init__(self, protein_structure_processor, initial_distance_matrix, structural_perturbation):
        self.processor = protein_structure_processor
        self.initial_distance_matrix = initial_distance_matrix
        
        # Apply structural perturbation and update distance matrix
        perturbation_effects = structural_perturbation.propagate_perturbation()
        self.current_distance_matrix = self.update_distance_matrix(perturbation_effects)
        
        self.distance_changes = {}
        self.functional_changes = {}  # Dictionary to store hypothetical or real functional changes

    def update_distance_matrix(self, perturbation_effects):
        """Update the current distance matrix based on the perturbation effects."""
        num_residues = len(self.initial_distance_matrix)
        updated_distance_matrix = np.copy(self.initial_distance_matrix)

        # Apply the perturbation effects to each distance
        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                effect_i = perturbation_effects.get(i, 0)
                effect_j = perturbation_effects.get(j, 0)
                perturbation_factor = 1 + (effect_i + effect_j) / 2  # Simple model to scale distance
                updated_distance_matrix[i][j] *= perturbation_factor
                updated_distance_matrix[j][i] = updated_distance_matrix[i][j]

        return updated_distance_matrix

    def calculate_distance_changes(self):
        """Calculate changes in distance between each pair of residues."""
        num_residues = len(self.initial_distance_matrix)
        epsilon = 1e-8  # Small value to avoid zero distance changes

        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                initial_distance = self.initial_distance_matrix[i][j]
                current_distance = self.current_distance_matrix[i][j]
                self.distance_changes[(i, j)] = (current_distance - initial_distance) + epsilon

    def set_functional_change(self, residue_index, change_value):
        """Set a functional change for a specific residue, if applicable."""
        self.functional_changes[residue_index] = change_value

    def calculate_correlation(self):
        """Calculate correlation between distance changes and functional changes."""
        distance_change_values = []
        functional_change_values = []

        # Collect corresponding distance changes and functional changes
        for (residue_pair, distance_change) in self.distance_changes.items():
            residue_index = residue_pair[0]  # Use the first residue in the pair to get functional change
            if residue_index in self.functional_changes and not np.isnan(distance_change):
                distance_change_values.append(distance_change)
                functional_change_values.append(self.functional_changes[residue_index])

        # Check for sufficient data points
        if len(distance_change_values) > 1:
            # Calculate correlation using numpy's corrcoef
            correlation = np.corrcoef(distance_change_values, functional_change_values)[0, 1]
            return correlation
        else:
            return None  # Not enough data points to calculate correlation

    def fenwick_tree_update(self, fenwick_tree, index, delta):
        """Update Fenwick Tree (Binary Indexed Tree) with a change in distance or function value."""
        while index < len(fenwick_tree):
            fenwick_tree[index] += delta
            index += index & -index

    def fenwick_tree_sum(self, fenwick_tree, index):
        """Calculate prefix sum up to a given index using Fenwick Tree (Binary Indexed Tree)."""
        sum_value = 0
        while index > 0:
            sum_value += fenwick_tree[index]
            index -= index & -index
        return sum_value

    def calculate_cumulative_distance_changes(self):
        """Calculate cumulative distance changes using a Fenwick Tree for efficient summation."""
        num_residues = len(self.processor.residues)
        fenwick_tree = [0] * (num_residues + 1)

        # Update Fenwick Tree with distance changes
        for (residue_pair, change) in self.distance_changes.items():
            residue_index = residue_pair[0]
            self.fenwick_tree_update(fenwick_tree, residue_index + 1, change)

        # Calculate cumulative changes
        cumulative_changes = {}
        for i in range(1, num_residues + 1):
            cumulative_changes[i - 1] = self.fenwick_tree_sum(fenwick_tree, i)

        return cumulative_changes


# Example usage
if __name__ == "__main__":
    pdb_file = "4WB7"  # Replace with your PDB file path
    perturbation_site = 398  # Site to apply perturbation

    # Initialize protein structure processor and calculate initial distance matrix
    protein_processor = ProteinStructureProcessor(pdb_file)
    initial_distance_matrix = protein_processor.distance_matrix

    # Initialize structural perturbation
    structural_perturbation = StructuralPerturbation(protein_processor, perturbation_site)
    structural_perturbation.compute_hessian_matrix()
    structural_perturbation.perform_eigen_decomposition()

    # Simulate correlation analysis with the perturbed structure
    correlation_analysis = CorrelationAnalysis(protein_processor, initial_distance_matrix, structural_perturbation)

    # Calculate distance changes after some hypothetical structural perturbation
    correlation_analysis.calculate_distance_changes()
    correlation_analysis.set_functional_change(118, 56)
    correlation_analysis.set_functional_change(375, 52)

    # Calculate correlation between distance changes and functional changes (if real data available)
    # Note: We don't set functional changes here since we are no longer using simulated data.

    # If you have real functional change data, set it here
    # correlation_analysis.set_functional_change(<residue_index>, <change_value>)

    # Calculate cumulative distance changes using Fenwick Tree
    cumulative_distance_changes = correlation_analysis.calculate_cumulative_distance_changes()
    print("Cumulative Distance Changes:", cumulative_distance_changes)
