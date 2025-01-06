import numpy as np
import heapq
from Bio import PDB
import matplotlib.pyplot as plt
import networkx as nx  # For visualization of pathways
from mpl_toolkits.mplot3d import Axes3D
from pdb_preprocessing import ProteinStructureProcessor
from energy_propagation import EnergyPropagation
from correlation_analysis import CorrelationAnalysis
from structural_perturbation import StructuralPerturbation

class ProteinAnalysisVisualizer:
    def __init__(self, processor, correlation_analysis):
        self.processor = processor
        self.correlation_analysis = correlation_analysis
        self.distance_changes = {}

    def get_correlated_residues(self, threshold=0.5):
        """Get residues with significant correlations above a given threshold."""
        correlated_residues = []

        for (residue_pair, distance_change) in self.correlation_analysis.distance_changes.items():
            residue_index = residue_pair[0]
            if residue_index in self.correlation_analysis.functional_changes:
                correlation = self.correlation_analysis.calculate_correlation()
                if correlation and abs(correlation) > threshold:
                    correlated_residues.append(residue_index)

        return correlated_residues

    def plot_protein_structure(self):
        """Visualize the 3D structure of the protein."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        coords = [residue['coordinates'] for residue in self.processor.residues]
        coords = np.array(coords)

        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='b', marker='o')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Protein Structure')
        plt.show()

    def plot_distance_changes(self):
        """Plot average distance changes per residue as a bar chart."""
        num_residues = len(self.processor.residues)
        residue_distance_changes = np.zeros(num_residues)

        # Calculate the total distance change for each residue by summing the pairwise changes
        for (i, j), change in self.correlation_analysis.distance_changes.items():
            residue_distance_changes[i] += abs(change)
            residue_distance_changes[j] += abs(change)

        # Average the distance change by dividing by the number of connections each residue has
        connections_count = np.zeros(num_residues)
        for (i, j) in self.correlation_analysis.distance_changes.keys():
            connections_count[i] += 1
            connections_count[j] += 1

        # Avoid division by zero by ensuring there are no residues with zero connections
        connections_count[connections_count == 0] = 1
        average_distance_changes = residue_distance_changes / connections_count

        # Plotting the average distance change for each residue
        plt.figure(figsize=(12, 6))
        plt.bar(range(num_residues), average_distance_changes, color='b')
        plt.xlabel('Residue Index')
        plt.ylabel('Average |Distance Change|')
        plt.title('Average Magnitude of Distance Changes for Each Residue')
        plt.show()

    def plot_correlation_analysis(self):
        """Plot the correlation between distance changes and functional changes."""
        correlation = self.correlation_analysis.calculate_correlation()
        if correlation is not None:
            plt.bar(['Correlation'], [correlation])
            plt.ylim([-1, 1])
            plt.title('Correlation Between Distance Changes and Functional Changes')
            plt.ylabel('Correlation Coefficient')
            plt.show()
        else:
            print("Not enough data points for correlation calculation.")

    def plot_correlated_residues_3d(self, correlated_residues):
        """Plot the correlated residues on the 3D structure."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all residues
        coords = np.array([residue['coordinates'] for residue in self.processor.residues])
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='b', marker='o', alpha=0.5, label='All Residues')

        # Check if correlated_residues is empty
        if len(correlated_residues) == 0:
            print("No correlated residues to plot.")
        else:
            # Plot correlated residues
            correlated_coords = np.array([self.processor.residues[i]['coordinates'] for i in correlated_residues])

            # Ensure correlated_coords is in the correct shape for plotting
            if correlated_coords.ndim == 1:
                correlated_coords = correlated_coords.reshape(1, -1)

            ax.scatter(correlated_coords[:, 0], correlated_coords[:, 1], correlated_coords[:, 2], c='r', marker='o', s=100, label='Correlated Residues')

        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Visualization of Correlated Residues')
        ax.legend()

        plt.show()

    def plot_cumulative_distance_changes(self):
        """Plot residues vs cumulative distance changes."""
        cumulative_changes = self.correlation_analysis.calculate_cumulative_distance_changes()
        residues = list(cumulative_changes.keys())
        changes = list(cumulative_changes.values())

        plt.figure(figsize=(10, 6))
        plt.bar(residues, changes, color='b')
        plt.xlabel('Residue Index')
        plt.ylabel('Cumulative Distance Change')
        plt.title('Residues vs Cumulative Distance Changes')
        plt.show()

    def plot_correlation_network_graph(self, correlated_residues):
        """Visualize correlated residues as a network graph."""
        G = nx.Graph()

        # Add nodes for correlated residues
        for residue in correlated_residues:
            G.add_node(residue, label=f"Residue {residue}")

        # Add edges between correlated residues if they are connected in the structure
        num_residues = len(self.processor.residues)
        for i in correlated_residues:
            for j in correlated_residues:
                if i != j and self.processor.adjacency_matrix[i][j] == 1:
                    G.add_edge(i, j)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='red', node_size=700, font_size=10)
        plt.title("Network of Correlated Residues")
        plt.show()

    def plot_3d_protein_heatmap(self):
        """Overlay a 3D heatmap of average distance changes on the 3D structure of the protein."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract coordinates and calculate average distance change per residue
        coords = np.array([residue['coordinates'] for residue in self.processor.residues])
        avg_distance_changes = np.zeros(len(self.processor.residues))
        connections_count = np.zeros(len(self.processor.residues))

        for (i, j), change in correlation_analysis.distance_changes.items():
            avg_distance_changes[i] += abs(change)
            avg_distance_changes[j] += abs(change)
            connections_count[i] += 1
            connections_count[j] += 1

        # Avoid division by zero
        connections_count[connections_count == 0] = 1
        avg_distance_changes /= connections_count

        # Plot the protein structure
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=avg_distance_changes, cmap='viridis', marker='o')
        fig.colorbar(sc, ax=ax, label='Average |Distance Change|')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Protein Structure with Heatmap of Average Distance Changes')

        plt.show()




# Example usage
if __name__ == "__main__":
    pdb_file = "4WB7"  # Replace with your PDB file path
    perturbation_site = 1  # Site to apply perturbation

    # Initialize protein structure processor and calculate initial distance matrix
    protein_processor = ProteinStructureProcessor(pdb_file)
    initial_distance_matrix = protein_processor.distance_matrix
    

    # Initialize structural perturbation
    structural_perturbation = StructuralPerturbation(protein_processor, perturbation_site)
    structural_perturbation.compute_hessian_matrix()
    structural_perturbation.perform_eigen_decomposition()
    explicit_perturbation_effects = structural_perturbation.explicit_perturbation_influence()
    print("Explicit Perturbation Effects:", explicit_perturbation_effects)


    # Simulate correlation analysis with the perturbed structure
    correlation_analysis = CorrelationAnalysis(protein_processor, initial_distance_matrix, structural_perturbation)
    correlation_analysis.set_functional_change(118, 56)
    correlation_analysis.set_functional_change(375, 52)


    # Calculate distance changes after some hypothetical structural perturbation
    correlation_analysis.calculate_distance_changes()
    cumulative_changes = correlation_analysis.calculate_cumulative_distance_changes()
    print("Cumulative Distance Changes:", cumulative_changes)

    # Create visualizer instance
    visualizer = ProteinAnalysisVisualizer(protein_processor, correlation_analysis)

    # Plot cumulative distance changes
    # visualizer.plot_cumulative_distance_changes()
    # visualizer.plot_distance_changes()

    # # Find residues with significant correlations
    # correlated_residues = visualizer.get_correlated_residues(threshold=0.5)

    # # Plot correlated residues in 3D structure
    # visualizer.plot_correlated_residues_3d(correlated_residues)

    # # Plot correlated residues as a network graph
    # visualizer.plot_correlation_network_graph(correlated_residues)

    # visualizer.plot_3d_protein_heatmap()





