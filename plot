import matplotlib.pyplot as plt
import numpy as np

layers_to_test = [2, 4, 8, 16]

results = {
    'GCN': {
        'acc': [0.7960, 0.7900, 0.5450, 0.1460],
        'energy': [3.06754, 17.42512, 4.15696, 0.03002]
    },
    'GIN': {
        'acc': [0.7320, 0.5580, 0.2080, 0.1910],
        'energy': [1104.56152, 580.61938, 57.50973, 0.13442]
    },
    'Transformer_NoPE': {
        'acc': [0.6440, 0.7210, 0.4810, 0.1490],
        'energy': [6.81688, 4.10833, 0.25067, 0.00001]
    },
    'Transformer_LapPE': {
        'acc': [0.7090, 0.7110, 0.1490, 0.1030],
        'energy': [4.30657, 3.56285, 0.00001, 0.00001]
    }
}

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for model_name, data_res in results.items():
    plt.plot(layers_to_test, data_res['acc'], marker='o', label=model_name)
plt.title('Test Accuracy vs. Network Depth')
plt.xlabel('Number of Layers (Depth)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
for model_name, data_res in results.items():
    plt.plot(layers_to_test, data_res['energy'], marker='x', label=model_name)
plt.title('Final Layer Dirichlet Energy vs. Depth')
plt.xlabel('Number of Layers (Depth)')
plt.ylabel('Dirichlet Energy (Log Scale)')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('grl_project_results.png')
print("Done!")
