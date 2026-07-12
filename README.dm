# Investigating Expressive Power and Information Propagation in GNNs

This repository contains the fully reproducible code-base for the Graph Representation Learning project. We analyze the properties of GCN, GIN, and Graph Transformers regarding representation collapse (oversmoothing) via node classification accuracy and Dirichlet energy constraints.

## Repository Structure
* `train_pipeline.py`: The core benchmarking pipeline implemented in PyTorch Geometric (PyG). It dynamically trains and evaluates GCN, GIN, and Graph Transformers across variable depths $T \in \{2, 4, 8, 16\}$ on the Cora dataset.
* `plot_results.py`: Visualization script extracting empirical evaluation metrics to synthesize accuracy trends and exponential Dirichlet energy decay graphs.

## Environmental Setup & Execution
To replicate the empirical results reported in the paper, install the dependencies and execute the scripts sequentially:

```bash
pip install torch torch-geometric scipy matplotlib numpy
python train_pipeline.py
python plot_results.py
