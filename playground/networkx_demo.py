import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_graph(nodes, edges, weights):
    G = nx.DiGraph()
    for i, node in enumerate(nodes):
        G.add_node(i, label=str(i))

    for (i, j), weight in zip(edges, weights):
        G.add_edge(i, j, weight=weight.item())  # Assume weight is a scalar for simplicity

    return G

def update_weights(weights):
    # This function would be called each epoch to update the weights in the graph
    for edge, weight in zip(model.edges, weights):
        G[edge[0]][edge[1]]['weight'] = weight.item()

def draw_graph(G):
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, 
            edge_color=[float(G[u][v]['weight']) for u, v in G.edges], 
            width=4.0, edge_cmap=plt.cm.Blues)

# Assuming edges and initial weights are defined
G = create_graph(model.nodes, model.edges, model.weights)

fig, ax = plt.subplots()

def animate(i):
    update_weights(model.weights)  # Fetch new weights
    ax.clear()
    draw_graph(G)
    ax.set_title(f"Epoch {i+1}")

ani = FuncAnimation(fig, animate, frames=epochs, repeat=True)
plt.show()
