from .TopolNet import TopolConv
import dgl
import torch as th

class TopolConvExplainer:
    def __init__(self, net):
        self.net = net
        # Extract net information.
        self.layer_params = []
        for layer in net.gcnlayers:
            # From source code: dgl/nn/pytorch/conv/graphconv.py
            W = layer.weight
            b = layer.bias
            # Done with source code.
            self.layer_params.append([b, W])

    def Validate(self, graph):
        A = graph.adj().to_dense()
        Dm = th.diag(th.pow(A.sum([1]), -0.5))
        Ak = Dm.matmul(A).matmul(Dm)
        hs = []
        h = graph.ndata[self.net.features_str].float()
        hs.append(h)
        for layer_param in self.layer_params:
            h = Ak.matmul(h).matmul(layer_param[1])+layer_param[0]
            h = h.clamp(min = 0)
            hs.append(h)
        return hs, Ak

    def LRP(self, graph, idxE):
        # Lambda function.
        rho = lambda W: W.abs()
        # The net and graph.
        num_layers = len(self.layer_params)
        R = [None]*(num_layers+1)
        num_nodes = graph.nodes().shape[0]        
        # The last layer.
        hs, Ak = self.Validate(graph)
        idxM = hs[-1][idxE].argmax()
        R[-1] = hs[-1][idxE][idxM]
        # The last-1 layer.
        rhoW = rho(self.layer_params[-1][1])
        R[-2] = Ak[idxE, :].unsqueeze(-1).matmul(rhoW[:, idxM].unsqueeze(0)) 
        z = R[-2].sum()
        R[-2] = R[-2]*(R[-1].item()/z)
        # The remaining layers.
        for L in range(num_layers-2, -1, -1):
            rhoW = rho(self.layer_params[L][1])            
            dimY, dimJ = rhoW.shape
            U = th.zeros(num_nodes, dimJ, num_nodes, dimY)
            for i in range(num_nodes):
                for j in range(dimJ):
                    U[i, j] = Ak[i, :].unsqueeze(-1).matmul(rhoW[:, j].unsqueeze(0))
            R[L] = th.einsum("ijxy,ij->xy", [U, R[L+1]/U.sum(dim=(2,3))])            

        return R

"""
    def SA(self, graph, idxE):
        num_nodes = graph.nodes().shape[0]
        hs, _ = self.Validate(graph)
        # SA
        dZ = 1
        t = hs[-1][idxE].argmax().item()
        sa = []
        for i in range(num_nodes):
            graph0 = graph            
            graph0.ndata[self.net.features_str][i] += dZ
            hs0, _ = self.Validate(graph0)
            t0 = hs0[-1][idxE].argmax().item()
            if t == t0:
                sa.append(-1)
            else:
                sa.append(t0 if i != idxE else -1)
            graph0.ndata[self.net.features_str][i] -= dZ
        # DA
        dZ = 1.
        da = []
        for i in range(num_nodes):
            graph0 = graph            
            graph0.ndata[self.net.features_str] = graph0.ndata[self.net.features_str].float()
            graph0.ndata[self.net.features_str][i] += dZ
            hs0, _ = self.Validate(graph0)
            dP = (hs0[-1][idxE][t]-hs[-1][idxE][t])/dZ
            da.append(dP.item() if i != idxE else 999)
            graph0.ndata[self.net.features_str][i] -= dZ
        return sa, da
"""

class TopolNetExplainer:
    def __init__(self, net):
        self.net = net
        self.topol_explainers = {element:TopolConvExplainer(net) for element, net in self.net.tclayers.items()}

    def LRP(self, graph):        
        num_atoms = graph.nodes().shape[0]
        Rs = []
        for idx_atom in range(num_atoms):            
            element = int(graph.ndata["features"][idx_atom])
            R = self.topol_explainers[element].LRP(graph, idx_atom)
            Ratom = [R0.sum(dim=1) for R0 in R[:-2]]
            Rs.append(Ratom[0])
        return Rs
