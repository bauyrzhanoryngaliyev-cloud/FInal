import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GINConv, TransformerConv
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse.linalg as sla
import numpy as np

# 1. Compute Laplacian Positional Encodings (LapPE) mathematically matching Section 2.1
def compute_laplacian_pe(data, k=8):
    num_nodes = data.num_nodes
    # Get the symmetric normalized Laplacian matrix
    L = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    
    # Eigen-decomposition fetching the smallest non-trivial eigenvectors
    try:
        eigenvalues, eigenvectors = sla.eigsh(L, k=k+1, which='SM', tol=1e-5)
    except:
        # Fallback to dense if sparse solver fails due to small graph structures
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Exclude the first trivial eigenvector (eigenvalue close to 0)
    pe = eigenvectors[:, 1:k+1]
    return torch.from_numpy(pe).float()

# 2. Dirichlet Energy Computation Utility matching Equation 1
def compute_dirichlet_energy(x, edge_index):
    src, dst = edge_index
    # Compute squared Euclidean distance across all connected edges
    edge_dist = torch.sum((x[src] - x[dst]) ** 2, dim=-1)
    # Normalize by 2 * |V| * d
    energy = torch.sum(edge_dist) / (2.0 * x.size(0) * x.size(1))
    return energy.item()

# 3. Model Architecture Implementations with Variable Depth T
class DeepGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, model_type='GCN'):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Input layer mapping
        if model_type == 'GCN':
            self.layers.append(GCNConv(in_channels, hidden_channels))
        elif model_type == 'GIN':
            self.layers.append(GINConv(nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )))
            
        # Hidden operational message-passing layers
        for _ in range(num_layers - 1):
            if model_type == 'GCN':
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            elif model_type == 'GIN':
                self.layers.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )))
                
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        final_embeddings = x
        out = self.classifier(final_embeddings)
        return out, final_embeddings

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_pe=False, pe_dim=8):
        super().__init__()
        self.use_pe = use_pe
        self.num_layers = num_layers
        
        if use_pe:
            self.pe_lin = nn.Linear(pe_dim, hidden_channels)
            
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # Global attention layers matching Transformer parameters
            self.layers.append(TransformerConv(hidden_channels, hidden_channels // 2, heads=2, dropout=0.1))
            
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, pe=None):
        h = self.input_lin(x)
        if self.use_pe and pe is not None:
            # Inject positional awareness via Equation 2
            h = h + self.pe_lin(pe)
            
        for i in range(self.num_layers):
            h = self.layers[i](h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)
            
        final_embeddings = h
        out = self.classifier(final_embeddings)
        return out, final_embeddings

# 4. Main Reproducible Pipeline
def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Cora dataset natively
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    
    # Precompute Laplacian Positional Encodings
    lap_pe = compute_laplacian_pe(data, k=8).to(device)
    
    depths = [2, 4, 8, 16]
    architectures = ['GCN', 'GIN', 'Transformer_NoPE', 'Transformer_LapPE']
    
    print("Executing architectural benchmarking pipeline...")
    
    for arch in architectures:
        for T in depths:
            # Instantiate model structures dynamically
            if arch == 'GCN':
                model = DeepGNN(dataset.num_features, 64, dataset.num_classes, T, 'GCN').to(device)
            elif arch == 'GIN':
                model = DeepGNN(dataset.num_features, 64, dataset.num_classes, T, 'GIN').to(device)
            elif arch == 'Transformer_NoPE':
                model = GraphTransformer(dataset.num_features, 64, dataset.num_classes, T, use_pe=False).to(device)
            elif arch == 'Transformer_LapPE':
                model = GraphTransformer(dataset.num_features, 64, dataset.num_classes, T, use_pe=True, pe_dim=8).to(device)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            
            # Standard optimization training execution loop
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                if 'Transformer' in arch:
                    out, embeddings = model(data.x, data.edge_index, lap_pe if 'LapPE' in arch else None)
                else:
                    out, embeddings = model(data.x, data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
            
            # Evaluation phase
            model.eval()
            with torch.no_grad():
                if 'Transformer' in arch:
                    out, embeddings = model(data.x, data.edge_index, lap_pe if 'LapPE' in arch else None)
                else:
                    out, embeddings = model(data.x, data.edge_index)
                    
                pred = out.argmax(dim=-1)
                correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                acc = int(correct) / int(data.test_mask.sum())
                
                # Capture information propagation state via Dirichlet Energy
                energy = compute_dirichlet_energy(embeddings, data.edge_index)
                
            print(f"Architecture: {arch:<18} | Depth T: {T:>2} | Test Accuracy: {acc:.4f} | Dirichlet Energy: {energy:.5f}")

if __name__ == '__main__':
    run_experiment()
