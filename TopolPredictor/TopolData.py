import torch as th
import dgl

class TopolData:
    def __init__(self):
        # General properties.
        self.features_str = "features"
        self.atomic_type_str = "atomic type"
        self.periodic_table = {"X":0, "H":1, "B":5, "C":6, "N":7, "O":8, "F":9,
            "P":15, "S":16, "CL":17, "AL":18,
            "SE":34, "BR":35, "I":53}
        self.num_features = 1
        # System-specified properties.
        self.atomic_types = {}  # All available atomic types.
        self.residues = []  # All residues: (graph, element indices, name)

    def ParseFromCHARMM(self, fn, device):
        with open(fn, "r") as fd:
            lines = fd.readlines()
            # Parse atomic types.
            atomic_types_numbers = [(line.split()[2].upper(), self.periodic_table[line.split()[4].upper()]) for line in lines if line.strip()[:4].upper() == "MASS"]
            self.atomic_types = { element:[atomic_types_number[0] for atomic_types_number in atomic_types_numbers if atomic_types_number[1] == element]
                for element in set([atomic_types_number[1] for atomic_types_number in atomic_types_numbers])}
            atomic_type_values = self.atomic_types.values()
            # Parse residues.
            self.residues = []
            MolTags = ["RESI", "PRES", "END"] # Indicate a molecule.
            BondTags = ["BOND", "DOUB", "TRIP"] # Indicate a bond.
            residue_name_lines = [(line_no, line.split()[1]) for line_no, line in enumerate(lines) if line.strip()[:4].upper() == "RESI"] # Only RESI is considered.
            for line_no, residue_name in residue_name_lines:
                # Collect residue lines.
                residue_lines = []
                idx = line_no                
                already_have_resi = False
                while True:
                    if lines[idx].strip()[:4].upper() in MolTags and already_have_resi:
                        break
                    if lines[idx].strip()[:4].upper() in MolTags and not already_have_resi:
                        already_have_resi = True
                    residue_lines.append(lines[idx])
                    idx += 1
                atom_lines = [atom[:atom.rfind("!")].strip() for atom in residue_lines if atom.strip()[:4].upper() == "ATOM"]
                bonding_lines = [bonding[:bonding.rfind("!")].strip() for bonding in residue_lines if bonding.strip()[:4].upper() in BondTags]
                # Parse the lines.
                atom_names = [line.split()[1].upper() for line in atom_lines]                
                vertices = [[(key, value.index(atom_type)) for key, value in self.atomic_types.items() if atom_type in value][0]
                    for atom_type in [line.split()[2].upper() for line in atom_lines]]
                edge_lines = []
                for line in bonding_lines:
                    edge_lines.extend(line.split()[1:])
                #print(residue_name, atom_names)
                edges1 = []
                edges2 = []
                for i in range(0, len(edge_lines), 2):
                    edges1.append(atom_names.index(edge_lines[i].upper()))
                    edges2.append(atom_names.index(edge_lines[i+1].upper()))
                    edges1.append(atom_names.index(edge_lines[i+1].upper()))
                    edges2.append(atom_names.index(edge_lines[i].upper()))
                # Transform to graph.
                graph = dgl.graph((th.tensor(edges1).to(device), th.tensor(edges2).to(device)))
                num_atoms = len(atom_names)
                # Add proterties.                
                graph.ndata[self.atomic_type_str] = th.LongTensor(num_atoms, 1).to(device)
                graph.ndata[self.features_str] = th.zeros(num_atoms, self.num_features).to(device)
                for i in range(num_atoms):
                    graph.ndata[self.atomic_type_str][i, 0] = vertices[i][1]
                    graph.ndata[self.features_str][i, 0] = vertices[i][0]
                # Add this graph.
                if 0 not in graph.in_degrees():
                    self.residues.append(
                        (graph,
                        {element:[idx for idx,value in enumerate(graph.ndata[self.features_str][:,0].tolist()) if value == element]
                            for element in set(graph.ndata[self.features_str][:,0].tolist())},
                        residue_name.upper())
                    )
