import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, TAGConv, SAGEConv, GATConv, DenseGraphConv, SGConv
import time
import tarfile, os, random

class TopolConv(nn.Module):
    def __init__(self, num_features, num_atomic_types, features_str):
        super(TopolConv, self).__init__()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Begin of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.gcnlayers = nn.ModuleList()
        num_hiddens = 16
        self.gcnlayers.append(GraphConv(num_features, num_hiddens,activation = nn.ReLU()))
        self.gcnlayers.append(GraphConv(num_hiddens, num_hiddens, activation = nn.ReLU()))
        self.gcnlayers.append(GraphConv(num_hiddens, num_atomic_types))
        self.fclayers = nn.ModuleList()        
        #self.fclayers.append(nn.Softmax(dim=1))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  End of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.features_str = features_str

    def forward(self, graph):
        h = graph.ndata[self.features_str]
        for layer in self.gcnlayers:
            h = layer(graph, h)
        for layer in self.fclayers:
            h = layer(h)
        return h

class TopolNet(nn.Module):
    def __init__(self, num_features, atomic_types, features_str):
        super(TopolNet, self).__init__()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Begin of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #        
        self.tclayers = {key:TopolConv(num_features, len(atomic_types[key]), features_str) for key in atomic_types}
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  End of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def forward(self, residue):
        graph = residue[0]        
        y = [0 for _ in range(len(graph.nodes()))]
        for key, value in residue[1].items():
            h = self.tclayers[key](graph)
            for i in value:
                y[i] = h[i]
        return y

    def save(self, prefix):
        with tarfile.open(prefix+".tar", "w") as tar:
            for key,value in self.tclayers.items():
                fn = str(key)+".pkl"
                th.save(value, fn)
                tar.add(fn)
                os.remove(fn)

    def load(self, prefix):
        with tarfile.open(prefix+".tar", "r") as tar:
            tar.extractall()
        for key, value in self.tclayers.items():
            fn = str(key)+".pkl"
            saved_net = th.load(fn, map_location = th.device('cpu'))
            os.remove(fn)
            # Below is a way to completely reload structure and parameters.
            self.tclayers[key] = saved_net
            # Below is a way to keep structure but change parameters.
            # value_dict = value.state_dict()
            # saved_dict = {k:v for k, v in saved_net.state_dict().items() if k in value_dict.keys()}
            # value_dict.update(saved_dict)
            # value.load_state_dict(value_dict)
    
    def to(self, device):
        for value in self.tclayers.values():
            value.to(device)
    
    def print(self):
        for key, value in self.tclayers.items():
            print("%s:" % (key))
            print(value)

def TopolTrain(net, residues, training_ratio, learning_rate, max_epochs, output_freq, save_fn_prefix, atomic_type_str, device):
    num_residues = len(residues)
    num_training_residues = int(num_residues*training_ratio)
    num_test_residues = num_residues-num_training_residues
    # Build the training data.    
    elements = list(net.tclayers.keys())
    num_elements = len(elements)
    random.shuffle(residues)
    training_residues = residues[:num_training_residues]
    test_residues = residues[num_training_residues:]
    training_labels = {element:th.cat([th.stack([residue[0].ndata[atomic_type_str][i] for i in residue[1][element]])
        for residue in training_residues if element in residue[1]]).squeeze(1).to(device) for element in elements}
    test_labels = {element:th.cat([th.stack([residue[0].ndata[atomic_type_str][i] for i in residue[1][element]])
        for residue in test_residues if element in residue[1]]).squeeze(1).to(device) for element in elements} if num_test_residues > 0 else None
    net.to(device)
    num_params = sum(sum(param.numel() for param in layer.parameters()) for layer in net.tclayers.values())
    optimizer = th.optim.Adam([{"params":layer.parameters()} for layer in net.tclayers.values()], lr = learning_rate, weight_decay = 1.e-4)
    criterion = nn.CrossEntropyLoss()
    # A closure to calculate loss.
    def CalcLoss(reses, labs):
        # Calculate loss function.
        output = {element:[] for element in elements}
        for res in reses:
            y = net(res)
            for key, value in res[1].items():
                output[key].extend([y[i] for i in value])  
        # Calculate correctness ratio.        
        num_labs = 0
        num_correct_labs = 0
        for element in elements:
            num_labs += len(labs[element])
            for i in range(len(labs[element])):
                if th.argmax(output[element][i]).item() == labs[element][i].item():
                    num_correct_labs += 1
        correctness = num_correct_labs/num_labs
        # Return.
        return sum(criterion(th.stack(output[element]), labs[element]) for element in elements)/num_elements, correctness
    # Print information.
    print(">>> Training of the Model >>>")
    print("Start at: ", time.asctime(time.localtime(time.time())))
    print("PID:      ", os.getpid())
    print("# of all graphs/labels:      %d" % (num_residues))
    print("# of training graphs/labels: %d" % (num_training_residues))
    print("# of test graphs/labels:     %d" % (num_test_residues))
    print("# of parameters:             %d" % (num_params))
    print("Learning rate:               %4.E" % (learning_rate))
    print("Maximum epochs:              %d" % (max_epochs))
    print("Output frequency:            %d" % (output_freq))
    print("Params filename prefix:      %s" % (save_fn_prefix))
    print("Device:                      %s" % (device))
    separator = "-"*100
    print(separator)
    print("%10s %15s %15s %15s   %s" % ("Epoch", "TrainingLoss", "TestLoss", "Time(s)", "SavePrefix"))
    print(separator)    
    # Train the model.
    t_begin = time.time()
    t0 = t_begin
    net.train()
    for epoch in range(max_epochs+1):
        # Calculate loss.
        current_training_loss = CalcLoss(training_residues, training_labels)[0]
        # Move forward.
        optimizer.zero_grad()
        current_training_loss.backward()        
        optimizer.step()
        # Output.
        if epoch % output_freq == 0:
            net.eval()
            training_correctness = CalcLoss(training_residues, training_labels)[1]
            test_correctness = CalcLoss(test_residues, test_labels)[1] if test_labels is not None else th.tensor([-1])
            net.train()
            prefix = save_fn_prefix+"-"+str(epoch)
            net.save(prefix)
            t1 = time.time()
            dur = t1-t0
            print("%10d %15.4f %15.4f %15.4f   %s" % (epoch, training_correctness, test_correctness, dur, prefix), flush = True)
            t0 = t1
    t_end = time.time()
    print(separator)
    print("Final loss: %.4f ppm" % (training_loss))
    print("Total training time: %.4f seconds" % (t_end-t_begin))
    print(">>> Training of the Model Accomplished! >>>")
    print("End at: ", time.asctime(time.localtime(time.time())))
