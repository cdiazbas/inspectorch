import zuko
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

# =============================================================================
class dot_dict(dict):
    """
    A dictionary subclass that allows for attribute-style access.

    This class extends the built-in dictionary to allow accessing keys as attributes.
    It overrides the __getattr__, __setattr__, and __delattr__ methods to provide
    this functionality.

    Example:
        d = DotDict({'key': 'value'})
        print(d.key)  # Outputs: value
        d.new_key = 'new_value'
        print(d['new_key'])  # Outputs: new_value
        del d.key
        print(d)  # Outputs: {'new_key': 'new_value'}
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# =============================================================================
class custom_dataset(torch.utils.data.Dataset):
    """
    A custom Dataset class for PyTorch.

    Attributes:
        lines (numpy.ndarray or pandas.DataFrame): The data to be used in the dataset.

    Methods:
        __init__(lines):
            Initializes the dataset with the given data.
        
        __len__():
            Returns the total number of samples in the dataset.
        
        __getitem__(index):
            Generates and returns one sample of data at the given index.
    """
    def __init__(self, lines):
        'Initialization'
        self.lines = lines

    def __len__(self):
        'Denotes the total number of samples'
        return self.lines.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.lines[index, :]

# =============================================================================
def create_flow(input_size=1, num_layers=5, hidden_features=32, num_bins=8, flow_type='NSF'):
    """
    Creates a flow model using Zuko's normalizing flows.

    Args:
        input_size (int): The size of the input features.
        num_layers (int): The number of flow layers to be used.
        hidden_features (int): The number of hidden features in each layer.
        num_bins (int): The number of bins for the spline transforms.
        flow_type (str): The type of flow to create ('NSF' or 'MAF').

    Returns:
        zuko.flows.Flow: A normalizing flow model.
    """
    if flow_type == 'NSF':
        flow = zuko.flows.NSF(
            features=input_size,
            context=0,  # Assuming no context is used
            transforms=num_layers,
            hidden_features=[hidden_features] * num_layers,
            bins=num_bins,
            activation=F.elu
        )
    elif flow_type == 'MAF':
        flow = zuko.flows.MAF(
            features=input_size,
            context=0,  # Assuming no context is used
            transforms=num_layers,
            hidden_features=[hidden_features] * num_layers,
            activation=F.elu
        )
    else:
        raise ValueError(f"Unsupported flow_type: {flow_type}")

    return flow


# =============================================================================
def print_summary(model):
    """
    Prints a summary of the model including the total number of parameters to optimize.

    Args:
        model (torch.nn.Module): The model to summarize.
    """
    pytorch_total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params to optimize:', pytorch_total_params_grad)

# =============================================================================
def train_flow(model, train_loader, learning_rate=1e-3, num_epochs=100, device='cpu'):
    """
    Trains a normalizing flow model using the provided training data loader.
    
    Args:
        model (zuko.flows.Flow): The normalizing flow model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.
        num_epochs (int, optional): Number of epochs to train the model. Default is 100.
        device (str, optional): Device to run the training on ('cpu' or 'cuda'). Default is 'cpu'.
    
    Returns:
        dict: A dictionary containing the trained model and the average training loss per epoch.
            - 'model' (zuko.flows.Flow): The trained model.
            - 'train_loss_avg' (list of float): List of average training losses for each epoch.
    """
    # We have to choose the optimization method and parameters to optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Normalize the data by the standard deviation:
    y_std = train_loader.dataset.lines.std(0).mean()
    y_mean = train_loader.dataset.lines.mean(0).mean()

    # Every loop takes a batch of data, calculates the logprob and change the 
    # parameters of the flow:
    train_loss_avg = []
    time0 = time.time()

    # Set the model in training mode
    model.train()

    # Move model parameters to the right device
    model = model.to(device)

    from tqdm import tqdm
    for epoch in range(1, num_epochs + 1):
        train_loss = []
        t = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (splines) in t:
            optimizer.zero_grad()

            y = (splines - y_mean)/y_std

            # In zuko, log_prob is a method directly on the flow object
            loss = -model().log_prob(y.to(device)).mean()

            loss.backward()
            optimizer.step()
            
            # Append the loss to the list
            train_loss.append(loss.item())

            # Update the progress bar
            t.set_postfix_str('Epoch: {0:2d}, Loss: {1:2.2f}'.format(epoch, train_loss[-1]))

        train_loss_avg.append(np.mean(np.array(train_loss)))

    # We can print the total training time:
    print('Training: {0:2.2f} min'.format((time.time()-time0)/60.))
    
    # Model to cpu again:
    model = model.to('cpu')
    
    # Return info in a dictionary:
    dict_info = {'model': model, 'train_loss_avg': train_loss_avg}
    
    return dict_info

# =================================================================
def nume2string(num):
    """ Convert number to scientific latex mode """
    mantissa, exp = f"{num:.2e}".split("e")
    return mantissa+ " \\times 10^{"+str(int(exp))+"}" 

# =============================================================================
def plot_train_loss(train_loss_avg):
    """
    Plots the training loss over the epochs.

    Args:
        train_loss_avg (list): The list of average training losses over the epochs.
    """
    fig = plt.figure()
    plt.plot(train_loss_avg, '.-')
    if len(train_loss_avg) > 1:
        output_title_latex = r'${:}$'.format(nume2string(train_loss_avg[-1]))
        plt.title('Final loss: '+output_title_latex)
    plt.minorticks_on()
    plt.xlabel('Epoch')
    plt.ylabel('Loss: -log prob')