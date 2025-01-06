from Bio import PDB

def extract_chain(pdb_id, chain_id):
    """
    Extracts a specific chain from a PDB file and writes it to a new file.
    
    Args:
        pdb_file: Path to the input PDB file.
        chain_id: The chain ID to extract (e.g., "A").
        output_file: Path to save the extracted chain.
    """
    pdb_file = f"{pdb_id}.pdb"
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    output_file = f"{pdb_id}_{chain_id}.pdb"
    
    # Create a new structure for the chain
    chain_structure = PDB.Structure.Structure("chain_" + chain_id)
    model = structure[0]  # Use the first model (most PDB files have only one model)
    chain = model[chain_id]  # Get the specific chain

    # Add the chain to the new structure
    new_model = PDB.Model.Model(0)
    new_model.add(chain.copy())
    chain_structure.add(new_model)

    # Save the chain to a new PDB file
    io = PDB.PDBIO()
    io.set_structure(chain_structure)
    io.save(output_file)
    print(f"Chain {chain_id} saved to {output_file}")

# Example usage
pdb_id = "4WB7"  
chain_id = "A" 
extract_chain(pdb_id, chain_id)
