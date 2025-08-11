import platform
import socket
import torch
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


def device_info():
    """
    Prints information about the current machine and available CUDA devices.

    If CUDA is available, this function prints the machine name, operating system,
    the number of available GPUs, and the name of each GPU. If CUDA is not available,
    it notifies the user that the CPU will be used instead.

    Requires:
        - torch
        - socket
        - platform
    """
    if torch.cuda.is_available():
        print(f"Machine name: {socket.gethostname()}")
        print(f"Operating system: {platform.system()} {platform.release()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")


def save_json(args, filename):
    """
    Saves the given arguments as a JSON file.

    Args:
        args (dict or iterable): The arguments to save. Should be convertible to a dictionary.
        filename (str): The path to the JSON file where the arguments will be saved.

    Notes:
        - If the directory for the given filename does not exist, it will be created.
        - The JSON file will be formatted with an indentation of 4 spaces.
    """
    import json

    args_dict = dict(args)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        json.dump(args_dict, f, indent=4)


def create_spectral_to_white_cmap():
    """
    Creates a colormap that uses the Red-Orange-Yellow transition from the
    'Spectral' colormap and then smoothly transitions from that yellow to
    white.
    """
    # 1. Get the original 'Spectral' colormap
    # We get it with 256 discrete colors
    spectral_map = plt.cm.get_cmap("Spectral", 256)

    # 2. Isolate the first half (Red -> Yellow)
    # The 'Spectral' map goes from Red to Yellow in its first half.
    # We take the first 128 colors to capture this transition.
    spectral_first_half = spectral_map(np.linspace(0, 0.45, 128))

    # 3. The endpoint of our first segment is the specific 'Spectral' yellow
    end_yellow = spectral_first_half[-1]

    # 4. Create the second half of the colormap: from our specific yellow to white
    # We create a small colormap segment for this transition, also with 128 steps.
    yellow_to_white_segment = LinearSegmentedColormap.from_list(
        "y2w_custom",
        [end_yellow, (1, 1, 1, 1)],  # (1,1,1,1) is RGBA for white
        N=128,
    )
    # Get the color values from this new segment
    yellow_to_white_colors = yellow_to_white_segment(np.arange(128))

    # 5. Combine the two lists of colors
    # We stack the first half on top of the second half.
    # We use np.vstack to join the two color arrays.
    final_colors = np.vstack((spectral_first_half, yellow_to_white_colors))

    # 6. Create the final, combined colormap
    custom_cmap = LinearSegmentedColormap.from_list("SpectralToWhite", final_colors)

    return custom_cmap


# Create our new colormap
spectral_to_white = create_spectral_to_white_cmap()
