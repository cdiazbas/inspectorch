# Inspectorch Examples

This folder contains examples and sample data to help you get started with `inspectorch`.

## Start here

**New to Inspectorch? Open [`01_normalizing_flow.ipynb`](01_normalizing_flow.ipynb).** It walks through the full workflow — loading the sample data, training a Normalizing Flow, and identifying rare events — and is the recommended starting point. If you prefer a plain script over a notebook, `01_normalizing_flow.py` covers the same steps.

Both use `hinode_data.npz`, a sample multidimensional solar dataset from the Hinode Spectro-Polarimeter included in this folder.

## Going further: Flow Matching

Once you are comfortable with the normalizing flow example, you can explore the Flow Matching backends, which are recommended for larger datasets or higher dimensions:

- `02_flow_matching.py` & `02_flow_matching.ipynb`: Same workflow as above, but using Flow Matching (Facebook's `flow_matching`) for density estimation.
- `03_flow_matching_cfm.ipynb`: Same example, but using the `torchcfm` API.

## Usage
You can run the Python scripts directly or open the Jupyter notebooks to interactively explore how Inspectorch processes the data, trains the models, and identifies rare events.
