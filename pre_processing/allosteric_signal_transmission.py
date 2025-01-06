import numpy as np
import heapq
from pdb_preprocessing import ProteinStructureProcessor

class AllostericSignalTransmission:
    def __init__(self, protein_structure_processor, covalent_cutoff=1.5, non_covalent_cutoff=8.0):
        """
        Initialize with covalent and non-covalent cutoff distances.
        - covalent_cutoff: distance threshold for covalent bonds
        - non_covalent_cutoff: distance threshold for non-covalent interactions
        """
        self.processor = protein_structure_processor
        self.covalent_cutoff = covalent_cutoff
        self.non_covalent_cutoff = non_covalent_cutoff
        self.covalent_interactions = {}
        self.non_covalent_interactions = {}

        # Classify interactions into covalent and non-covalent
        self.classify_interactions()

    def classify_interactions(self):
        """Classify interactions between residues as covalent or non-covalent."""
        num_residues = len(self.processor.residues)
        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                distance = self.processor.distance_matrix[i][j]
                if distance <= self.covalent_cutoff:
                    self.covalent_interactions[(i, j)] = distance
                    self.covalent_interactions[(j, i)] = distance
                elif distance <= self.non_covalent_cutoff:
                    self.non_covalent_interactions[(i, j)] = distance
                    self.non_covalent_interactions[(j, i)] = distance

    def propagate_signal_through_covalent(self, start_residue):
        """Propagate signal through covalent interactions using a priority queue."""
        pq = [(0, start_residue)]
        visited = {start_residue: 0}
        
        while pq:
            current_signal, residue = heapq.heappop(pq)
            for (res1, res2), distance in self.covalent_interactions.items():
                if res1 == residue:
                    new_signal = current_signal + (1 / distance)
                    if res2 not in visited or new_signal < visited[res2]:
                        visited[res2] = new_signal
                        heapq.heappush(pq, (new_signal, res2))
        return visited

    def propagate_signal_through_non_covalent(self, start_residue):
        """Propagate signal through non-covalent interactions with a different decay rate."""
        pq = [(0, start_residue)]
        visited = {start_residue: 0}
        
        while pq:
            current_signal, residue = heapq.heappop(pq)
            for (res1, res2), distance in self.non_covalent_interactions.items():
                if res1 == residue:
                    decay_factor = np.exp(-distance)  # Decay signal exponentially over non-covalent bonds
                    new_signal = current_signal + decay_factor
                    if res2 not in visited or new_signal < visited[res2]:
                        visited[res2] = new_signal
                        heapq.heappush(pq, (new_signal, res2))
        return visited


# Example usage
if __name__ == "__main__":
    pdb_id = "4WB7" 
    allosteric_site = 391
    
    # Initialize protein structure
    protein_processor = ProteinStructureProcessor(pdb_id)

    # Initialize and classify covalent and non-covalent interactions
    signal_transmission = AllostericSignalTransmission(protein_processor)

    # Propagate signal through covalent bonds
    covalent_signal_propagation = signal_transmission.propagate_signal_through_covalent(allosteric_site)
    print("Signal Propagation through Covalent Interactions:", covalent_signal_propagation)

    # Propagate signal through non-covalent bonds
    non_covalent_signal_propagation = signal_transmission.propagate_signal_through_non_covalent(allosteric_site)
    print("Signal Propagation through Non-Covalent Interactions:", non_covalent_signal_propagation)