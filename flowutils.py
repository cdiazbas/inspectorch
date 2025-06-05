import torch
import nflows
from nflows import transforms
from nflows.transforms import CompositeTransform
from nflows import utils
from nflows.nn import nets
import torch.nn.functional as F
import numpy as np
from nflows.flows.base import Flow
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
def piecewise_rational_quadratic_coupling_transform(iflow, input_size, hidden_size, num_blocks=1, activation=F.elu, num_bins=8):
    """
    Creates a Piecewise Rational Quadratic Coupling Transform for use in normalizing flows.

    This function constructs a coupling transform based on a piecewise rational quadratic spline,
    which is parameterized by a neural network. The coupling transform splits the input into two
    parts, applies a transformation to one part conditioned on the other, and combines them back.

    Args:
        iflow (int): Index of the flow, used to determine the binary mask pattern.
        input_size (int): The size of the input features.
        hidden_size (int): The number of hidden units in each layer of the neural network.
        num_blocks (int, optional): The number of residual blocks in the neural network. Defaults to 1.
        activation (callable, optional): The activation function to use in the neural network. Defaults to F.elu.
        num_bins (int, optional): The number of bins for the piecewise rational quadratic spline. Defaults to 8.

    Returns:
        transforms.PiecewiseRationalQuadraticCouplingTransform: A coupling transform object configured
        with the specified parameters.
    """
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(input_size, even=(iflow % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
            in_features=in_features, 
            out_features=out_features, 
            hidden_features=hidden_size, 
            num_blocks=num_blocks,
            activation=activation
        ),
        num_bins=num_bins, 
        tails='linear', 
        tail_bound=5, 
        apply_unconditional_transform=False
    )
    
# =============================================================================
def masked_piecewise_rational_quadratic_autoregressive_transform(input_size, hidden_size, num_blocks=1, activation=F.elu, num_bins=8):
    return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=input_size,
        hidden_features=hidden_size,
        num_bins=num_bins,
        tails='linear',
        tail_bound=6,
        use_residual_blocks=True,
        random_mask=False,
        activation=activation,
        num_blocks=num_blocks,
    )

# =============================================================================
def masked_umnn_autoregressive_transform(input_size, hidden_size, num_blocks=1, activation=F.elu):
    from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
    return MaskedUMNNAutoregressiveTransform(features=input_size,
        hidden_features=hidden_size,
        use_residual_blocks=True,
        random_mask=False,
        activation=activation,
        num_blocks=num_blocks,
    )

# =============================================================================
def create_linear_transform(param_dim):
    """
    Creates a composite linear transformation consisting of a random permutation 
    of features followed by an LU linear transformation.

    Args:
        param_dim (int): The dimensionality of the input features.

    Returns:
        transforms.CompositeTransform: A composite transform that applies a 
        random permutation and an LU linear transformation to the input features.
    """
    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True)
    ])
        
# =============================================================================
def create_flow(input_size=1, num_layers=5, hidden_features=32,num_bins=8, flow_type='PRQCT'):
    """
    Creates a flow model using Piecewise Rational Quadratic Coupling Transforms and linear transforms.

    Args:
        input_size (int, optional): The size of the input features. Default is 1.
        num_layers (int, optional): The number of flow layers to be used. Default is 5.
        hidden_features (int, optional): The number of hidden features in the coupling transform's neural network. Default is 32.

    Returns:
        CompositeTransform: A composite transform consisting of alternating linear and coupling transforms.
    """
    base_dist = nflows.distributions.StandardNormal((input_size,))

    transformsi = []
    for i in range(num_layers):
        transformsi.append(create_linear_transform(param_dim=input_size))
        transformsi.append(piecewise_rational_quadratic_coupling_transform(i, input_size, hidden_features,num_bins=num_bins))
    transformsi.append(create_linear_transform(param_dim=input_size))
    
    transformflow = CompositeTransform(transformsi)
    
    return Flow(transformflow, base_dist)

# =============================================================================
def create_flow_autoregressive(input_size=1, num_layers=5, hidden_features=32,num_bins=8):
    """
    Creates a flow model using Piecewise Rational Quadratic Coupling Transforms and linear transforms.

    Args:
        input_size (int, optional): The size of the input features. Default is 1.
        num_layers (int, optional): The number of flow layers to be used. Default is 5.
        hidden_features (int, optional): The number of hidden features in the coupling transform's neural network. Default is 32.

    Returns:
        CompositeTransform: A composite transform consisting of alternating linear and coupling transforms.
    """
    base_dist = nflows.distributions.StandardNormal((input_size,))

    transformsi = []
    for i in range(num_layers):
        transformsi.append(create_linear_transform(param_dim=input_size))
        transformsi.append(masked_umnn_autoregressive_transform(input_size, hidden_features))
        # transformsi.append(masked_piecewise_rational_quadratic_autoregressive_transform(input_size, hidden_features,num_bins=num_bins))
    transformsi.append(create_linear_transform(param_dim=input_size))
    
    transformflow = CompositeTransform(transformsi)
    
    return Flow(transformflow, base_dist)

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
        model (torch.nn.Module): The normalizing flow model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.
        num_epochs (int, optional): Number of epochs to train the model. Default is 100.
        device (str, optional): Device to run the training on ('cpu' or 'cuda'). Default is 'cpu'.
    Returns:
        dict: A dictionary containing the trained model and the average training loss per epoch.
            - 'model' (torch.nn.Module): The trained model.
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
    model.to(device)

    from tqdm import tqdm
    for epoch in range(1, num_epochs + 1):
        train_loss = []
        t = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (splines) in t:
            optimizer.zero_grad()

            y = (splines - y_mean)/y_std

            loss = -model.log_prob(inputs=y.to(device)).mean()

            loss.backward()
            optimizer.step()
            
            # Append the loss to the list
            train_loss.append(loss.item())

            # Update the progress bar
            t.set_postfix_str('Epoch: {0:2d}, Loss: {1:2.2f}'.format(epoch, train_loss[-1]))

        train_loss_avg.append(np.mean(np.array(train_loss)))

    # We can print the total training time:
    print('Training: {0:2.2f} min'.format( (time.time()-time0)/60.) )
    
    # Model to cpu again:
    model.to('cpu')
    
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