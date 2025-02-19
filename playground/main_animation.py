from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from matplotlib.animation import PillowWriter  # For GIF saving
from matplotlib.animation import FFMpegWriter  # For video saving

class Node(nn.Module):
    def __init__(self, *layers):
        super(Node, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Model(nn.Module):
    def __init__(self, nodes: list[Node], edges: list[tuple[int, int]]):
        super(Model, self).__init__()
        self.nodes = nn.ModuleList(nodes)
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in edges])
        self.connections = {i: [] for i in range(len(nodes))}
        for idx, (i, j) in enumerate(edges):
            self.connections[i].append((j, self.weights[idx]))

    def forward(self, x: torch.Tensor):
        buffer = [None] * (len(self.nodes) + 1)
        buffer[0] = x
        for i, node in enumerate(self.nodes):
            y = node(buffer[i]) if buffer[i] is not None else None
            for j, w in self.connections.get(i, []):
                if buffer[j] is None:
                    buffer[j] = y * w
                else:
                    buffer[j] += y * w
        return buffer[-1]

def create_graph(nodes, edges, weights):
    G = nx.DiGraph()
    for i in range(len(nodes)):
        G.add_node(i, label=str(i))
    for (i, j), weight in zip(edges, weights):
        G.add_edge(i, j, weight=weight)  # `weight` is already a float, no `.item()` needed
    return G

def draw_graph(G, pos, weights, ax):
    ax.clear()
    edge_vmin = min(weights)
    edge_vmax = max(weights)
    edge_colors = weights
    edges = nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                                   width=4, edge_cmap=plt.cm.Blues, edge_vmin=edge_vmin, edge_vmax=edge_vmax)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=700)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={node: node for node in G.nodes()})
    ax.set_title("Graph Structure")
    return edges, edge_vmin, edge_vmax

# Main execution
if __name__ == "__main__":
    # Data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Model definition
    model = Model([
        Node(nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2)),
        Node(nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2)),
        Node(nn.Flatten()),
        Node(nn.Linear(1600, 256), nn.ReLU(), nn.Linear(256, 128), nn.Sigmoid()),
        Node(nn.Linear(1600, 256), nn.ReLU()),
        Node(nn.Linear(128, 10)),
        Node(nn.Linear(256, 10))
    ], [(0,1), (1,2), (2,3), (2,4), (3,5), (4,6), (5,7), (6,7)]).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    weights_history = []

    # Training process
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Collect weights for animation
        weights_history.append([w.detach().cpu().item() for w in model.weights])
        print(f"Epoch {epoch+1} completed.")

    # Initialization of the figure, graph, and colorbar
    fig, ax = plt.subplots()
    G = create_graph(model.nodes, [(0,1), (1,2), (2,3), (2,4), (3,5), (4,6), (5,7), (6,7)], weights_history[0])
    pos = nx.spring_layout(G)
    edges, vmin, vmax = draw_graph(G, pos, weights_history[0], ax)

    # Create a ScalarMappable for color bar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=plt.cm.Blues)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Edge Weight')

    def update(frame):
        weights = weights_history[frame]
        edges, vmin, vmax = draw_graph(G, pos, weights, ax)
        sm.set_array([])
        sm.set_clim(vmin, vmax)
        cbar.update_normal(sm)
        return edges, cbar

    ani = FuncAnimation(fig, update, frames=len(weights_history), repeat=False)
    plt.show()

    # After setting up the animation with FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(weights_history), repeat=False)

    # Specify the writer
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Save as MP4
    #ani.save('my_animation.mp4', writer=writer)

    # Alternatively, to save as a GIF
    writer_gif = PillowWriter(fps=15) 
    ani.save('my_animation.gif', writer=writer_gif)