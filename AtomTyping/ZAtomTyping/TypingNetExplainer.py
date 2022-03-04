from .TypingNet import TypingConv
import dgl
import torch as th

class TypingConvExplainer:
    def __init__(self, net):
        self.net = net
        # Extract net information.
        self.maxK = 0
        self.layer_params = []
        for layer in net.gcnlayers:
            # From source code: dgl/nn/pytorch/conv/tagconv.py
            W = layer.lin.weight
            b = layer.lin.bias
            K = layer._k
            # Done with source code.
            dW = W.shape[1]//(K+1)    
            Wk = [W[:, k*dW:(k+1)*dW].t() for k in range(K+1)]            
            self.layer_params.append([b, K, Wk])
            self.maxK = K if K > self.maxK else self.maxK

    def Validate(self, graph):
        Dm = th.diag(th.pow(graph.in_degrees().float().clamp(min=1), -0.5))
        A = Dm.matmul(graph.adj().to_dense()).matmul(Dm)
        Ak = [th.matrix_power(A, k) for k in range(self.maxK+1)]        
        hs = []
        h = graph.ndata[self.net.features_str].float()
        hs.append(h)
        for layer_param in self.layer_params:
            h = sum(Ak[k].matmul(h).matmul(layer_param[2][k]) for k in range(layer_param[1]+1))+layer_param[0]
            h = h.clamp(min = 0)
            hs.append(h)
        return hs, Ak

    def LRP(self, graph, idxE):
        # Lambda function.
        rho = lambda Wk: [W.abs() for W in Wk]
        # The net and graph.
        num_layers = len(self.layer_params)
        R = [None]*(num_layers+1)
        num_nodes = graph.nodes().shape[0]        
        # The last layer.
        hs, Ak = self.Validate(graph)
        idxM = hs[-1][idxE].argmax()
        R[-1] = hs[-1][idxE][idxM]
        # The last-1 layer.
        rhoWk = rho(self.layer_params[-1][2])
        K = self.layer_params[-1][1]
        R[-2] = sum(Ak[k][idxE, :].unsqueeze(-1).matmul(rhoWk[k][:, idxM].unsqueeze(0)) for k in range(K+1))
        z = R[-2].sum()
        R[-2] = R[-2]*(R[-1].item()/z)
        # The remaining layers.
        for L in range(num_layers-2, -1, -1):
            rhoWk = rho(self.layer_params[L][2])
            K = self.layer_params[L][1]
            dimY, dimJ = rhoWk[0].shape
            U = th.zeros(num_nodes, dimJ, num_nodes, dimY)
            for i in range(num_nodes):
                for j in range(dimJ):
                    U[i, j] = sum(Ak[k][i, :].unsqueeze(-1).matmul(rhoWk[k][:, j].unsqueeze(0)) for k in range(K+1))
            R[L] = th.einsum("ijxy,ij->xy", [U, R[L+1]/U.sum(dim=(2,3))])            

        return R

class TypingNetExplainer:
    def __init__(self, net):
        self.net = net
        self.typing_explainers = {element:TypingConvExplainer(net) for element, net in self.net.tclayers.items()}

    def LRP(self, graph):        
        num_atoms = graph.nodes().shape[0]
        Rs = []
        for idx_atom in range(num_atoms):            
            element = int(graph.ndata["features"][idx_atom])
            R = self.typing_explainers[element].LRP(graph, idx_atom)
            Ratom = [R0.sum(dim=1) for R0 in R[:-2]]
            Rs.append(Ratom[0])
        return Rs
