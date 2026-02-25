# Inspectorch Examples

This folder contains examples and sample data to help you get started with `inspectorch`.

## Contents

If you want to use the Normalizing Flow backend for high precision density estimation, you can use the following examples:

- `hinode_data.npz`: A sample multidimensional solar dataset from the Hinode Spectro-Polarimeter, used in the examples.
- `example_normalizing_flow.py` & `example_normalizing_flow.ipynb`: Examples demonstrating how to use the Normalizing Flow backend for high precision density estimation.

If you prefer to explore the experimental implementation using Flow Matching, you can use the following examples: 
- `example_flow_matching.py` & `example_flow_matching.ipynb`: Examples demonstrating how to use Flow Matching (Facebook's `flow_matching`) for density estimation.
- `example_flow_matching_cfm.ipynb`: Same examples as above, but using the `torchcfm` API.

## Usage
You can run the Python scripts directly or open the Jupyter notebooks to interactively explore how Inspectorch processes the data, trains the models, and identifies rare events.

