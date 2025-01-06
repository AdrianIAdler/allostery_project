import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pdb_preprocessing import ProteinStructureProcessor
from energy_propagation import EnergyPropagation



class AllostericAnalysis:
    def __init__(self, energy_propagation, active_site_index):
        """
        Analyze energy propagation results with respect to each residue's allosteric connection to the active site.
        Args:
            energy_propagation: Instance of EnergyPropagation with calculated energy propagation data
            active_site_index: Index of the active site residue in the protein structure
        """
        self.energy_propagation = energy_propagation.get_energy_propagation()
        self.active_site_index = active_site_index
        self.residue_influence_scores = {}

    def analyze_residue_influences(self):
        """
        Analyze how each residue propagates energy or connects to the active site.
        The idea is to check the energy value associated with each residue's influence on the active site.
        """
        # Analyze how connected/propagated each residue's energy is to the active site
        for residue_index, energy_value in self.energy_propagation.items():
            # The closer (or higher the energy signal) this value, the stronger the connection
            self.residue_influence_scores[residue_index] = abs(energy_value)

        # Sort the residues by their influence (descending order)
        sorted_influence = dict(sorted(self.residue_influence_scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_influence

    def get_influence_scores(self):
        """
        Returns the computed influence scores of residues with respect to their connection to the active site.
        """
        print({"SCORES":self.residue_influence_scores})
        return self.residue_influence_scores


class VisualizationManager:
    def __init__(self, analysis_results, protein_processor):
        """
        Manages visualization of the analyzed data.
        Args:
            analysis_results: Results from AllostericAnalysis
            protein_processor: An instance of ProteinStructureProcessor to map residues to their structure
        """
        self.analysis_results = analysis_results
        self.processor = protein_processor

    def plot_3d_heatmap(self):
        """Visualize the influence scores across residues as a heatmap overlaid on the 3D structure of the protein."""
        residue_positions = self.processor.get_residue_positions()
        influence_scores = np.array([self.analysis_results.get(i, 0) for i in range(len(self.processor.residues))])
        normalized_scores = (influence_scores - influence_scores.min()) / (influence_scores.max() - influence_scores.min() + 1e-5)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = zip(*residue_positions)
        scatter = ax.scatter(x, y, z, c=normalized_scores, cmap="viridis", s=50)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label("Influence Score", rotation=270, labelpad=15)
        ax.set_title("3D Heatmap of Residue Influence Scores on Protein Structure")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def plot_top_residues(self, top_n=100):
        """Plot the top N residues influencing the active site connection."""
        sorted_scores = dict(sorted(self.analysis_results.items(), key=lambda item: item[1], reverse=True))
        top_residues = list(sorted_scores.items())[:top_n]
        residues = [item[0] for item in top_residues]
        scores = [item[1] for item in top_residues]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=residues, y=scores, palette="viridis")
        plt.title("Top Residues Influencing the Active Site")
        plt.xlabel("Residue Index")
        plt.ylabel("Influence Score")
        plt.xticks(rotation=90)  
        plt.tight_layout()  
        plt.show()

    def plot_residue_scores(self):
        """Plot all residues against their influence scores with horizontal connections."""
        residue_indices = list(self.analysis_results.keys())
        scores = list(self.analysis_results.values())

        plt.figure(figsize=(12, 6))
        # sns.barplot(x=residue_indices, y=scores, palette="viridis")
        plt.scatter(residue_indices, scores, linestyle="-", color="red", label="Residues") 
        plt.title("All Residues vs Influence Scores")
        plt.xlabel("Residue Index")
        plt.ylabel("Influence Score")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=90)  
        plt.tight_layout()  
        plt.show()



    def plot_network_graph(self):
        """
        Create a network graph showing how residues influence the active site communication pathways.
        Uses matplotlib's network plotting utilities.
        """
        
        # Create a graph from the analysis results
        G = nx.Graph()

        threshold = 0.1  # Define a threshold for including edges to simplify the network visualization
        for residue_index, score in self.analysis_results.items():
            if score >= threshold:
                G.add_node(residue_index)
                for other_residue_index in self.analysis_results.keys():
                    if residue_index != other_residue_index and abs(self.analysis_results[residue_index] - self.analysis_results[other_residue_index]) < threshold:
                        G.add_edge(residue_index, other_residue_index)

        # Draw the graph
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)  # NetworkX layout
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="blue")
        plt.title("Network Graph of Residues and Active Site Influence")
        plt.show()


if __name__ == "__main__":
    # Load protein structure
    pdb_id = "4WB7"
    allosteric_site_index = 0  # Index of the allosteric site
    active_site_index = 398  # Replace with index of the active site residue

    # Initialize ProteinProcessor
    protein_processor = ProteinStructureProcessor(pdb_id, chain_id="A")

    # Perform energy propagation
    energy_propagation = EnergyPropagation(protein_processor, allosteric_site_index, protein_processor.bond_strengths)
    energy_propagation.propagate_energy()
    propagation_results = energy_propagation.get_energy_propagation()

    # Perform analysis of propagation results
    analysis = AllostericAnalysis(energy_propagation, active_site_index)
    influence_scores = analysis.analyze_residue_influences()
    print(analysis.get_influence_scores)
    # Visualize the results
    viz_manager = VisualizationManager(influence_scores, protein_processor)
   
    viz_manager.plot_top_residues(top_n=100)
    viz_manager.plot_network_graph()
    viz_manager.plot_residue_scores()
    viz_manager.plot_3d_heatmap

