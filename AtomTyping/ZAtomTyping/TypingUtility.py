from re import L
from .TypingData import TypingData
from .TypingNet import TypingNet, TypingTrain
import torch as th
import dgl

class TypingUtility:
    def __init__(self):
        self.fn = None
        self.data = None
        self.net = None
        self.prob = th.nn.Softmax(dim=0)

    def Predict(self, fn):
        residue = self.DLText2Residue(fn)
        return residue, self.PredictResidue(residue)
    
    def PredictResidue(self, residue):
        with th.no_grad():
            y = self.net(residue)
            types = [self.data.atomic_types[residue[0].ndata[self.data.features_str][i][0].item()][th.argmax(y0).item()] for i, y0 in enumerate(y)]
            probs = [self.prob(y0) for y0 in y]
            return types, probs

    def DLText2Residue(self, fn):        
        with open(fn, "r") as fd:
            lines = fd.readlines()
            atom_numbers = [int(word) for word in lines[0].split()]
            edge_lines = [int(word) for word in lines[1].split()]
            edges1 = []
            edges2 = []
            for i in range(0, len(edge_lines), 2):
                edges1.append(edge_lines[i])
                edges2.append(edge_lines[i+1])
                edges1.append(edge_lines[i+1])
                edges2.append(edge_lines[i])
            graph = dgl.graph((th.tensor(edges1), th.tensor(edges2)))            
            graph.ndata[self.data.features_str] = th.tensor(atom_numbers).unsqueeze(1)            
            return (graph,
                {element:[idx for idx,value in enumerate(graph.ndata[self.data.features_str][:, 0].tolist()) if value == element]
                            for element in set(graph.ndata[self.data.features_str][:, 0].tolist())},
                fn.upper())
    
    def Calibrate(self):
        type_stats = {key:[[0,0] for _ in range(len(value))] for key,value in self.data.atomic_types.items()}
        for residue in self.data.residues:
            y = self.net(residue)
            for element, type, prob in zip(residue[0].ndata[self.data.features_str][:, 0].tolist(), residue[0].ndata[self.data.atomic_type_str].squeeze(1).tolist(), y):
                type_stats[element][type][0] += 1                
                predict_type = th.argmax(prob).item()
                if predict_type != type:
                    type_stats[element][type][1] += 1
                    if self.data.atomic_types[element][type] == "CG321":
                        print("In %8s: should be %8s, but is %s" % (residue[2], self.data.atomic_types[element][type], self.data.atomic_types[element][predict_type]))
        return type_stats
    
    def ExportParams(self, fn):        
        with open(fn, "w") as fd:
            maxK = -1
            num_layers = len(self.net.tclayers[1].gcnlayers)
            layer_str = ""
            for i in range(num_layers-1): layer_str += "layer%d, " % (i)
            layer_str += "layer%d" % (num_layers-1)        
            fd.write("Layer %s;\n" % (layer_str))
            for element, net in self.net.tclayers.items():            
                params = net.export()
                param_str = ""
                for i, param in enumerate(params):
                    b = param[0]
                    K = param[1]
                    maxK = max(K, maxK)
                    Ws = param[2]
                    dim0 = Ws[0].shape[0]
                    dim1 = Ws[0].shape[1]
                    param_str += "layer%d = {{" % (i)
                    param_str += ("MatrixXd::Zero(%d, %d), " % (dim0, dim1))*(K+1)
                    param_str += "}, RowVectorXd::Zero(%d)};\n" % (dim1)
                    # Write down parameters.
                    for k in range(K+1):
                        param_str += "layer%d.Ws[%d] << " % (i, k)
                        for x in range(dim0-1):
                            for y in range(dim1):
                                param_str += "%15.8f, " % (Ws[k][x][y])
                            param_str += "\n"
                        for y in range(dim1-1): param_str += "%15.8f, " % (Ws[k][dim0-1][y])
                        param_str += "%15.8f;\n" % (Ws[k][dim0-1][dim1-1])
                    param_str += "layer%d.b << " % (i)
                    for y in range(dim1-1): param_str += "%15.8f, " % (b[y])
                    param_str += "%15.8f;\n" % (b[dim1-1])
                param_str += "Nets[%d] = {%s};" % (element, layer_str)
                fd.write("%s\n" % (param_str))
            fd.write("MaxK = %d;" % (maxK))
 
    def Build(self, fn, params_fn = None, type = "CHARMM", training_ratio = 1., learning_rate = 1E-3, max_epochs = 1000000, output_freq = 100, device = th.device("cpu")):
        print(" -- Build Atom Typing Predictor --")        
        # Build typing model.
        self.fn = fn
        self.data = TypingData()
        typing_type_fun = {"CHARMM":self.data.ParseFromCHARMM}
        typing_type_fun[type.upper()](self.fn, device)
        print("Definitions from:  %s" % (self.fn))
        num_atomic_types = len(self.data.atomic_types)
        print(" Atomic types: %d" % (sum(len(value) for value in self.data.atomic_types.values())))
        for key, value in self.data.atomic_types.items():
            print("  %-2s: %-3d" % ([k for k, v in self.data.periodic_table.items() if v == key][0], len(value)), value)
        print(" Residue names: %d" % (len(self.data.residues)))
        # Build net.
        print("Net parameters:   %s" % ("To be trained" if params_fn is None else params_fn))
        num_features = self.data.num_features
        features_str = self.data.features_str
        atomic_type_str = self.data.atomic_type_str
        save_fn_prefix = self.fn[:self.fn.rfind(".")]        
        self.net = TypingNet(num_features, self.data.atomic_types, features_str)
        if params_fn is not None:
            # Only load parameters, not architecture.            
            self.net.load(params_fn)
        if max_epochs > 0:
            TypingTrain(self.net, self.data.residues, training_ratio, learning_rate, max_epochs, output_freq, save_fn_prefix, atomic_type_str, device)
        self.net.eval()
        print(" -- Building Atom Typing Predictor Accomplished --")
