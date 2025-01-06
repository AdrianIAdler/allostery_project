import numpy as np
import heapq
from Bio import PDB
import requests
from communication_pathways import CommunicationPathways
from pdb_preprocessing import ProteinStructureProcessor
from energy_propagation import EnergyPropagation

class StructuralPerturbation:
    def __init__(self, protein_structure_processor, perturbation_site):
        self.processor = protein_structure_processor
        self.perturbation_site = perturbation_site
        self.hessian_matrix = None  # Hessian matrix for second derivatives
        self.eigenvalues = None     # Eigenvalues of the Hessian matrix
        self.eigenvectors = None    # Eigenvectors for normal mode analysis

    def compute_hessian_matrix(self):
        """Compute the Hessian matrix (second derivative of energy) for the protein structure."""
        num_residues = len(self.processor.residues)
        self.hessian_matrix = np.zeros((num_residues, num_residues))

        # Compute second derivatives based on distances
        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                distance = self.processor.distance_matrix[i][j]
                if distance > 0:
                    # Second derivative example based on distance
                    self.hessian_matrix[i][j] = -1 / distance**2
                    self.hessian_matrix[j][i] = self.hessian_matrix[i][j]

        # Set diagonal elements for self-interaction
        for i in range(num_residues):
            self.hessian_matrix[i][i] = -np.sum(self.hessian_matrix[i])

    def perform_eigen_decomposition(self):
        """Calculate eigenvalues and eigenvectors for normal mode analysis."""
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hessian_matrix)

    def propagate_perturbation(self):
        """Propagate a structural perturbation using normal mode analysis."""
        if self.hessian_matrix is None:
            self.compute_hessian_matrix()
        if self.eigenvalues is None or self.eigenvectors is None:
            self.perform_eigen_decomposition()

        # Propagate perturbation along low-frequency modes
        perturbation_effect = {}
        for i in range(len(self.processor.residues)):
            effect = 0
            for mode in range(len(self.eigenvalues)):
                # Low-frequency (large eigenvalues) modes propagate the perturbation more strongly
                if self.eigenvalues[mode] > 0:
                    contribution = (self.eigenvectors[i, mode] ** 2) / self.eigenvalues[mode]
                    effect += contribution
            perturbation_effect[i] = effect
        return perturbation_effect
    
    def explicit_perturbation_influence(self):
        """Explicitly incorporate the perturbation site into the propagation calculations."""
        if self.hessian_matrix is None:
            self.compute_hessian_matrix()
        if self.eigenvalues is None or self.eigenvectors is None:
            self.perform_eigen_decomposition()

        # Propagate perturbation explicitly from the perturbation site
        perturbation_effect = {}
        for i in range(len(self.processor.residues)):
            effect = 0
            for mode in range(len(self.eigenvalues)):
                # Low-frequency (large eigenvalues) modes propagate the perturbation more strongly
                if self.eigenvalues[mode] > 0:
                    contribution = (self.eigenvectors[self.perturbation_site, mode] * self.eigenvectors[i, mode]) / self.eigenvalues[mode]
                    effect += contribution
            perturbation_effect[i] = effect
        return perturbation_effect


# Example usage
if __name__ == "__main__":
    pdb_id = "4WB7"  # Replace with your PDB file path
    perturbation_site = 398  # Site to apply perturbation

    # Initialize protein structure and energy propagation
    protein_processor = ProteinStructureProcessor(pdb_id)
    # energy_propagation = EnergyPropagation(protein_processor, allosteric_site, protein_processor.bond_strengths)
    # energy_propagation.propagate_energy()

    # Initialize structural perturbation
    structural_perturbation = StructuralPerturbation(protein_processor, perturbation_site)

    # Compute Hessian matrix and perform eigen decomposition
    structural_perturbation.compute_hessian_matrix()
    structural_perturbation.perform_eigen_decomposition()

    # Propagate perturbation and get results
    perturbation_effects = structural_perturbation.propagate_perturbation()
    print("Perturbation Effects:", perturbation_effects)

    # Propagate perturbation explicitly from the perturbation site and get results
    explicit_perturbation_effects = structural_perturbation.explicit_perturbation_influence()
    print("Explicit Perturbation Effects:", explicit_perturbation_effects)
