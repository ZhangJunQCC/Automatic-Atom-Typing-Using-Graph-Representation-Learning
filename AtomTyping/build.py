from ZAtomTyping import TypingNet, TypingUtility, TypingNetExplainer
import torch as th

if __name__=='__main__':
    # Set up options.
    root_dir = "TrainedModels/"
    typing_fn = root_dir+"top_all36_cgenff-sub.rtf"
    params_fn = None
    training_ratio = 0.8
    learning_rate = 5.E-3
    max_epochs = 12000  # Set a negative one, we will use it directly.
    output_freq = 1
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # Build the typing model.
    typing_utility = TypingUtility()
    typing_utility.Build(typing_fn,
        params_fn = params_fn,
        training_ratio = training_ratio,
        learning_rate = learning_rate,
        max_epochs = max_epochs,
        output_freq = output_freq,
        device = device)
    print("")
