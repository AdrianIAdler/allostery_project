import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pdb_preprocessing import ProteinStructureProcessor
from communication_pathways import CommunicationPathways
from energy_propagation import EnergyPropagation

class CommunicationPathwaysAnalysis:
    def __init__(self, communication_pathways, active_site_index):
        """
        Analyze shortest communication pathways between residues.

        Args:
            communication_pathways: Instance of CommunicationPathways with calculated shortest path data
            active_site_index: Index of the active site residue in the protein structure
        """
        self.communication_pathways = communication_pathways
        self.active_site_index = active_site_index
        self.pathway_scores = {}

    def analyze_pathway_lengths(self, allosteric_site):
        """
        Analyze the shortest path lengths from the allosteric site to all other residues.

        Args:
            allosteric_site: Index of the residue acting as the allosteric site

        Returns:
            A dictionary of residue indices and their shortest path lengths to the active site
        """
        shortest_paths = self.communication_pathways.get_all_paths_from_allosteric_site(allosteric_site)
        for residue_index, path_length in shortest_paths.items():
            self.pathway_scores[residue_index] = path_length

        # Sort the residues by their shortest path lengths (ascending order)
        sorted_pathways = dict(sorted(self.pathway_scores.items(), key=lambda item: item[1]))
        return sorted_pathways

    def get_pathway_scores(self):
        """
        Returns the computed shortest pathway scores for residues.
        """
        return self.pathway_scores

class VisualizationManager:
    def __init__(self, analysis_results, protein_processor):
        """
        Manages visualization of the analyzed data.

        Args:
            analysis_results: Results from CommunicationPathwaysAnalysis
            protein_processor: An instance of ProteinStructureProcessor to map residues to their structure
        """
        self.analysis_results = analysis_results
        self.processor = protein_processor

    def plot_3d_heatmap(self):
        """Visualize the shortest pathway lengths as a heatmap overlaid on the 3D structure of the protein."""
        residue_positions = self.processor.get_residue_positions()
        pathway_scores = np.array([self.analysis_results.get(i, float('inf')) for i in range(len(self.processor.residues))])
        normalized_scores = (pathway_scores - pathway_scores.min()) / (pathway_scores.max() - pathway_scores.min() + 1e-5)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = zip(*residue_positions)
        scatter = ax.scatter(x, y, z, c=normalized_scores, cmap="viridis", s=50)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label("Pathway Length", rotation=270, labelpad=15)
        ax.set_title("3D Heatmap of Residue Pathway Lengths")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def plot_top_residues(self, top_n=100):
        """Plot the top N residues with shortest pathways to the active site."""
        sorted_scores = dict(sorted(self.analysis_results.items(), key=lambda item: item[1]))
        top_residues = list(sorted_scores.items())[:top_n]
        residues = [item[0] for item in top_residues]
        scores = [item[1] for item in top_residues]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=residues, y=scores, palette="viridis")
        plt.title("Top Residues with Shortest Pathways to the Active Site")
        plt.xlabel("Residue Index")
        plt.ylabel("Shortest Path Length")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_network_graph(self):
        """Create a network graph showing the shortest communication pathways."""
        G = nx.Graph()

        threshold = np.percentile(list(self.analysis_results.values()), 90)  # Include top 10% of residues by pathway length
        for residue_index, score in self.analysis_results.items():
            if score <= threshold:
                G.add_node(residue_index)
                for neighbor_index in range(len(self.processor.residues)):
                    if self.processor.adjacency_matrix[residue_index][neighbor_index] == 1:
                        G.add_edge(residue_index, neighbor_index)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="blue")
        plt.title("Network Graph of Communication Pathways")
        plt.show()

if __name__ == "__main__":
    # Load protein structure and define bond strengths
    pdb_id = "4WB7"  # Replace with your PDB file
    allosteric_site = 119  # Index of the residue acting as the allosteric site
    active_site = 398      # Example active site residue index

    # Initialize the processor and energy propagation classes
    protein_processor = ProteinStructureProcessor(pdb_id, chain_id= "A")
    energy_propagation = EnergyPropagation(protein_processor, allosteric_site, protein_processor.bond_strengths)
    energy_propagation.propagate_energy()

    # Initialize communication pathway analysis
    pathways = CommunicationPathways(protein_processor, energy_propagation)

    # Get the shortest communication pathway from the allosteric site to the active site
    shortest_path_distance = pathways.get_shortest_path(allosteric_site, active_site)
    # print(f"Shortest path from allosteric site {allosteric_site} to active site {active_site}: {shortest_path_distance}")

    # Get shortest paths from allosteric site to all residues
    all_paths_from_allosteric_site = pathways.get_all_paths_from_allosteric_site(allosteric_site)
    # print("Shortest paths from allosteric site to all residues:", all_paths_from_allosteric_site)

    # # Perform communication pathway analysis
    # pathways = CommunicationPathways(protein_processor, None)
    analysis = CommunicationPathwaysAnalysis(pathways, active_site)
    pathway_scores = analysis.analyze_pathway_lengths(allosteric_site)

    # Visualize the results
    viz_manager = VisualizationManager(pathway_scores, protein_processor)
    viz_manager.plot_top_residues(top_n=100)
    viz_manager.plot_network_graph()
    viz_manager.plot_3d_heatmap()
