"""import torch
import torch.utils.data
import flowutils

...

# data.shape -> (PIXELS, CHANNELS)

# Data container:
args = flowutils.dot_dict()
args.batch_size = 1000

training_set = flowutils.custom_dataset(data)
train_loader = torch.utils.data.DataLoader(
    training_set, batch_size=args.batch_size, shuffle=True
)

# We normalize the data (only internally) as NFlows performs better with normalized data
y_std = train_loader.dataset.lines.std((0))
y_mean = train_loader.dataset.lines.mean((0))

...

# The Flow class creates the flow model by joining the base distribution and the previously defined layers.
model = flowutils.create_flow(
    input_size=CHANNELS, num_layers=20, hidden_features=64, num_bins=8
)

args.learning_rate = 5e-4
args.num_epochs = 10

dict_info = flowutils.train_flow(
    model,
    train_loader,
    learning_rate=args.learning_rate,
    num_epochs=args.num_epochs,
    device=device,
)

flowutils.plot_train_loss(dict_info["train_loss_avg"])


# Evaluate all the dataset and get the log probability of each point
log_prob = (
    model.log_prob(inputs=(torch.from_numpy(data) - y_mean) / y_std).detach().numpy()
)

# Reshape log_prob to match the original data dimensions
log_prob_reshaped = rearrange(log_prob, "(y x) -> y x", y=ny, x=nx)
"""
