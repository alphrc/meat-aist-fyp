from __future__ import annotations

import copy
import random
from functools import reduce
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from concurrent.futures import ThreadPoolExecutor


class Utils:
    @staticmethod
    def _create_graph(gene: Gene) -> dict[int, list[int]]:
        graph = defaultdict(list)
        for i, j in gene.edges:
            graph[i].append(j)
        return graph
    
    @staticmethod
    def _topological_sort_util(gene: Gene, v: str, visited: set, stack: deque):
        visited.add(v)
        for i in [edge[1] for edge in gene.edges if edge[0] == v]:
            if i not in visited:
                Utils._topological_sort_util(gene, i, visited, stack)
        stack.appendleft(v)

    @staticmethod
    def _topological_sort(gene: Gene):
        visited = set()
        stack = deque()
        for i in gene.nodes:
            if i not in visited:
                Utils._topological_sort_util(gene, i, visited, stack)
        return list(stack)
    
    @staticmethod
    def _dfs(gene: Gene, start_node: str, current_node: str, reach_map: dict[str, set], visited: set):
        for edge in gene.edges:
            if edge[0] == current_node and edge[1] not in visited:
                visited.add(edge[1])
                reach_map[start_node].add(edge[1])
                Utils._dfs(gene, start_node, edge[1], reach_map, visited)

    @staticmethod
    def _create_reachability_map(gene: Gene) -> dict[int, set[int]]:
        reachability_map = {i: set([i]) for i in gene.nodes.keys()}
        for node in gene.nodes.keys():
            visited = set()
            Utils._dfs(gene, node, node, reachability_map, visited)
        return reachability_map

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)

    def forward(self, data: torch.Tensor):
        data = data.permute(2, 0, 1)
        attn_output, attn_output_weights = self.mha(data, data, data)
        attn_output = attn_output.permute(1, 2, 0)
        return attn_output

class Node(nn.Module):
    def __init__(self, *layers, in_shape: tuple = None, out_shape: tuple = None, out_spatial: bool = None):
        super(Node, self).__init__()
        self.layers = nn.Sequential(*layers)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.out_spatial = out_spatial

    def forward(self, *args):
        if self.layers is None:
            return args
        return self.layers(*args)

class NodeID(str):
    def __new__(cls, value):
        return super(NodeID, cls).__new__(cls, value)

class NodeFactory:
    def __init__(self):
        self.handlers = {
            (True, 3, 3): self._handle_spatial_3_3,
            (True, 2, 2): self._handle_spatial_2_2,
            (True, 1, 1): self._handle_spatial_1_1,
            (True, 3, 1): self._handle_spatial_n_1,
            (True, 2, 1): self._handle_spatial_n_1,
            (False, 3, 1): self._handle_nonspatial_n_1,
            (False, 2, 1): self._handle_nonspatial_n_1,
            (False, 1, 1): self._handle_nonspatial_1_1,
        }

class Gene:
    def __init__(self, nodes: dict[NodeID, Node], edges: list[tuple[NodeID, NodeID]], ancestors: list[Gene] = None):
        self.nodes = nodes
        self.edges = edges
        self.ancestors = ancestors
        self._initialize()

    def _initialize(self) -> None:
        self.weights = {edge: nn.Parameter(torch.ones(1)) for edge in self.edges}
        self.graph = Utils._create_graph(self)
        self.sorted_nodes = Utils._topological_sort(self)
        self.reachability_map = Utils._create_reachability_map(self)

    def _new_id(self, offset: int = 0) -> NodeID:
        int_ids = [int(id) for id in self.nodes.keys() if id.isnumeric()]
        new_id = NodeID(max(int_ids) + offset + 1) if int_ids else NodeID(offset)
        return new_id

    def _replicate(self) -> Gene:
        return Gene(copy.deepcopy(self.nodes), copy.deepcopy(self.edges), ancestors=self)

    def _random_new_node_by_shape(self, in_shape, out_shape: tuple, in_spatial: bool) -> Node: # [TODO] Add more cases
        if in_spatial:
            if len(in_shape) == 3 and len(out_shape) == 3:
                if in_shape[1:] == out_shape[1:]:
                    candidates = [
                        Node(nn.Conv2d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        Node(nn.Conv2d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.Sigmoid(), in_shape=in_shape, out_shape=out_shape, out_spatial=True)
                    ]
                elif in_shape[1]/out_shape[1] == in_shape[2]/out_shape[2] == int(in_shape[1]/out_shape[1]):
                    size = int(in_shape[1]/out_shape[1])
                    candidates = [
                        Node(nn.MaxPool2d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        Node(nn.AvgPool2d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        Node(nn.ReLU(), nn.MaxPool2d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        Node(nn.ReLU(), nn.AvgPool2d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        Node(nn.Conv2d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.ReLU(), nn.MaxPool2d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        Node(nn.Conv2d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.ReLU(), nn.AvgPool2d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                    ]
                else: return None
            elif len(in_shape) == 2 and len(out_shape) == 2:
                if in_shape == out_shape:
                    candidates = [
                        Node(nn.Conv1d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        Node(SelfAttention(in_shape[0], 1), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                    ]
                elif in_shape[1] == out_shape[1]:
                    candidates = [
                        Node(nn.Conv1d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                    ]
                elif in_shape[1]/out_shape[1] == int(in_shape[1]/out_shape[1]):
                        size = int(in_shape[1]/out_shape[1])
                        candidates = [
                            Node(nn.MaxPool1d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                            Node(nn.AvgPool1d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                            Node(nn.Conv1d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.ReLU(), nn.MaxPool1d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                            Node(nn.Conv1d(in_shape[0], out_shape[0], 3, 1, 'same'), nn.ReLU(), nn.AvgPool1d(size), in_shape=in_shape, out_shape=out_shape, out_spatial=True),
                        ]
                else: return None
            elif len(in_shape) == 1 and len(out_shape) == 1:
                if in_shape == out_shape:
                    candidates = [
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=False),
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.Sigmoid(), in_shape=in_shape, out_shape=out_shape, out_spatial=False)
                    ]
                else:
                    candidates = [
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=False),
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.Sigmoid(), in_shape=in_shape, out_shape=out_shape, out_spatial=False)
                    ]
            elif len(in_shape) > 1 and len(out_shape) == 1:
                product = reduce(lambda x, y: x * y, in_shape)
                candidates = [
                    Node(nn.Flatten(), nn.Linear(product, out_shape[0]), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=False),
                    Node(nn.Flatten(), nn.Linear(product, out_shape[0]), nn.Sigmoid(), in_shape=in_shape, out_shape=out_shape, out_spatial=False)
                ]
        else:
            if len(in_shape) == 1 and len(out_shape) == 1:
                if in_shape == out_shape:
                    candidates = [
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=False),
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.Sigmoid(), in_shape=in_shape, out_shape=out_shape, out_spatial=False)
                    ]
                else:
                    candidates = [
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=False),
                        Node(nn.Linear(in_shape[0], out_shape[0]), nn.Sigmoid(), in_shape=in_shape, out_shape=out_shape, out_spatial=False)
                    ]
            elif len(in_shape) > 1 and len(out_shape) == 1:
                product = reduce(lambda x, y: x * y, in_shape)
                candidates = [
                    Node(nn.Flatten(), nn.Linear(product, out_shape[0]), nn.ReLU(), in_shape=in_shape, out_shape=out_shape, out_spatial=False),
                    Node(nn.Flatten(), nn.Linear(product, out_shape[0]), nn.Sigmoid(), in_shape=in_shape, out_shape=out_shape, out_spatial=False)
                ]
            else: return None
        choice = random.choice(candidates)
        return choice

    def _random_mid_shape(self, in_shape: tuple, out_shape: tuple, in_spatial: bool) -> tuple:
        candidate_shapes = [in_shape, out_shape]

        if len(in_shape) == 1 and len(out_shape) == 1:
            candidate_shapes.extend([(in_shape[0] * 2,), (in_shape[0] * 4,)])
            candidate_shapes.extend([(out_shape[0] * 2,), (out_shape[0] * 4,)])
        if len(in_shape) == 2 and len(out_shape) == 1:
            candidate_shapes.append((in_shape[0] * 2, in_shape[1])) # For 1d conv with more channels
            candidate_shapes.append((in_shape[0] * in_shape[1],))
            candidate_shapes.extend([(out_shape[0] * 2,), (out_shape[0] * 4,)])
        if len(in_shape) == 3 and len(out_shape) == 1:
            candidate_shapes.append((in_shape[0] * 2, in_shape[1], in_shape[2])) # For 2d conv with more channels
            candidate_shapes.append((in_shape[0] * in_shape[1] * in_shape[2],))
            candidate_shapes.extend([(out_shape[0] * 2,), (out_shape[0] * 4,)])

        return random.choice(candidate_shapes)

    def _is_valid_new_edge(self, i: NodeID, j: NodeID) -> bool:
        if i == j:
            return False
        if self.nodes[i].out_shape is None or self.nodes[j].in_shape is None:
            return False
        if i in self.reachability_map[j]:
            return False
        if len(self.nodes[i].out_shape) < len(self.nodes[j].in_shape):
            return False
        if not self.nodes[i].out_spatial and self.nodes[j].out_spatial:
            return False
        return  True
    
    def _is_same_shape(self, i: NodeID, j: NodeID) -> bool:
        return self.nodes[i].out_shape == self.nodes[j].in_shape

    def _mutate_add_one_node(self, in_node: NodeID = None, out_node: NodeID = None) -> Gene:
        new_gene = self._replicate()
        candidate_edges = [(i, j) for i in self.nodes.keys() for j in self.nodes.keys() if self._is_valid_new_edge(i, j)]

        if not candidate_edges:
            return new_gene

        while True:
            in_node, out_node = random.choice(candidate_edges)
            in_shape, out_shape, in_spatial = self.nodes[in_node].out_shape, self.nodes[out_node].in_shape, self.nodes[in_node].out_spatial

            new_node = self._random_new_node_by_shape(in_shape, out_shape, in_spatial)
            
            if new_node is not None:
                break

        new_id = self._new_id()
        new_gene.nodes[new_id] = new_node
        new_gene.edges.append((in_node, new_id))
        new_gene.edges.append((new_id, out_node))

        new_gene._initialize()

        return new_gene

    def _mutate_add_two_node(self, in_node: NodeID = None, out_node: NodeID = None) -> Gene:
        new_gene = self._replicate()
        candidate_edges = [(i, j) for i in self.nodes.keys() for j in self.nodes.keys() if self._is_valid_new_edge(i, j)]

        if not candidate_edges:
            return new_gene
        
        if in_node is not None and out_node is not None and (in_node, out_node) in candidate_edges:
            candidate_edges = [(in_node, out_node)]
        else:
            return new_gene

        while True:
            in_node, out_node = random.choice(candidate_edges)
            in_shape, out_shape, in_spatial = self.nodes[in_node].out_shape, self.nodes[out_node].in_shape, self.nodes[in_node].out_spatial
            mid_shape = self._random_mid_shape(in_shape, out_shape, in_spatial)

            new_node_1 = self._random_new_node_by_shape(in_shape, mid_shape, in_spatial)
            new_node_2 = self._random_new_node_by_shape(mid_shape, out_shape, in_spatial)

            if new_node_1 is not None and new_node_2 is not None:
                break

        new_id_1 = self._new_id()
        new_id_2 = self._new_id(1)
        new_gene.nodes[new_id_1] = new_node_1
        new_gene.nodes[new_id_2] = new_node_2
        new_gene.edges.append((in_node, new_id_1))
        new_gene.edges.append((new_id_1, new_id_2))
        new_gene.edges.append((new_id_2, out_node))

        new_gene._initialize()

        return new_gene

    def _mutate_replicate_node(self) -> Gene:
        new_gene = self._replicate()
        candidate_ids = [id for id in self.nodes.keys() if id not in [NodeID('in'), NodeID('out')]]

        if not candidate_ids:
            return new_gene

        target_id = random.choice(candidate_ids)
        new_id = self._new_id()

        new_gene.nodes[new_id] = copy.deepcopy(self.nodes[target_id])
        new_gene.edges.extend([(new_id, i) for i in self.graph[target_id]])
        new_gene.edges.extend([(i, new_id) for i in self.graph if target_id in self.graph[i]])

        new_gene._initialize()

        return new_gene
    
    def _add_edge(self) -> Gene:
        new_gene = self._replicate()
        candidate_edges = [(i, j) for i in self.nodes.keys() for j in self.nodes.keys() if self._is_valid_new_edge(i, j) and self._is_same_shape(i, j)]

        if not candidate_edges:
            return new_gene

        in_node, out_node = random.choice(candidate_edges)
        new_gene.edges.append((in_node, out_node))
        new_gene._initialize()

        return new_gene

    def _remove_edge(self) -> Gene:
        new_gene = self._replicate()
        candidate_edges = [edge for edge in self.edges if torch.abs(self.weights[edge]) < 0.1]

        if not candidate_edges:
            return new_gene

        in_node, out_node = random.choice(candidate_edges)
        new_gene.edges.remove((in_node, out_node))
        new_gene._initialize()

        return new_gene

    def mutate(self) -> Gene:
        method = [
            self._mutate_add_one_node,
            self._mutate_add_two_node,
            self._mutate_replicate_node,
            self._add_edge,
            self._remove_edge
        ]
        return random.choice(method)()
    
    def crossover(self, other: Gene) -> Gene:
        pass

class Population:
    def __init__(self, gene_pool: list[Gene], capacity: int = 16):
        self.gene_pool = gene_pool

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.epochs = 4

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    @staticmethod
    def train_loop(dataloader: DataLoader, model: Model, loss_fn: nn.CrossEntropyLoss, optimizer: optim.SGD, device):
        total_loss = 0
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return total_loss / len(dataloader)

    def _evaluate(self, gene) -> tuple[Gene, float]:
        model = Model(gene).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        final_loss = 0
        for epoch in range(self.epochs):
            #print(f"Epoch {epoch+1}\n-------------------------------")
            final_loss = Population.train_loop(self.train_loader, model, loss_fn, optimizer, self.device)
        print("Done!")
        return (gene, final_loss)

    def evaluate(self):
        with ThreadPoolExecutor(max_workers=16) as executor:  # Limit to a sensible number of threads
            results = list(executor.map(self._evaluate, self.gene_pool))
        self.generation_losses = results

    def select(self):
        self.generation_losses.sort(key=lambda x: x[1])
        self.gene_pool = [gene_loss_pair[0] for gene_loss_pair in self.generation_losses[:8]]

    def evolve(self):
        print(f"Before: {len(self.gene_pool)}")
        self.evaluate()
        self.select()
        print(f"Before Loss: {[l for g, l in self.generation_losses]}")
        for i, (gene, loss) in enumerate(self.generation_losses):
            print(f"Model {i}: Loss {loss} Nodes {gene.nodes} Edges {gene.edges}")

        new_gene_pool = []
        for gene in self.gene_pool:
            new_gene_pool.append(gene.mutate())
            new_gene_pool.append(gene.mutate())
            new_gene_pool.append(gene.mutate())
        self.gene_pool.extend(new_gene_pool)
        print(f"After: {len(self.gene_pool)}")

class Model(nn.Module):
    def __init__(self, gene: Gene):
        super(Model, self).__init__()
        self.gene = gene
        self.nodes = nn.ModuleDict(copy.copy(gene.nodes))
        self.weights = nn.ParameterList(gene.weights.values())

    def forward(self, x: torch.Tensor):
        buffer = {id: None for id in self.gene.nodes.keys()}
        buffer['in'] = x

        for id in self.gene.sorted_nodes[:-1]:
            y = self.nodes[id](buffer[id]) if buffer[id] is not None else None
            for j in self.gene.graph.get(id, []):
                w = self.gene.weights[(id, j)]
                buffer[j] = buffer[j] + y * w if buffer[j] is not None else y * w
        return buffer['out']


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    gene = Gene(
        nodes={
            'in': Node(in_shape=None, out_shape=(1, 28, 28), out_spatial=True),
            '0': Node(nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2), in_shape=(1, 28, 28), out_shape=(32, 13, 13), out_spatial=True),
            '1': Node(nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2), in_shape=(32, 13, 13), out_shape=(64, 5, 5), out_spatial=True),
            '2': Node(nn.Flatten(), in_shape=(64, 5, 5), out_shape=(1600,), out_spatial=False),
            '3': Node(nn.Linear(1600, 256), nn.ReLU(), nn.Linear(256, 128), nn.Sigmoid(), in_shape=(1600,), out_shape=(128,), out_spatial=False),
            '4': Node(nn.Linear(1600, 256), nn.ReLU(), in_shape=(1600,), out_shape=(256,), out_spatial=False),
            '5': Node(nn.Linear(128, 10), in_shape=(128,), out_shape=(10,), out_spatial=False),
            '6': Node(nn.Linear(256, 10), in_shape=(256,), out_shape=(10,), out_spatial=False),
            'out': Node(in_shape=(10,), out_shape=None, out_spatial=False)
        }, edges=[
            ('in', '0'),
            ('0', '1'),
            ('1', '2'),
            ('2', '3'),
            ('2', '4'),
            ('3', '5'),
            ('4', '6'),
            ('5', 'out'),
            ('6', 'out')
        ]
    )

    genome = Gene(
        nodes = {
            'in': Node(in_shape=None, out_shape=(1, 28, 28), out_spatial=True),
            '0': Node(nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2), in_shape=(1, 28, 28), out_shape=(32, 13, 13), out_spatial=True),
            '1': Node(nn.Flatten(), nn.Linear(5408, 10), in_shape=(1, 28, 28), out_shape=(10,), out_spatial=False),
            'out': Node(in_shape=(10,), out_shape=None, out_spatial=False)
        }, edges = [
            ('in', 'out')
        ]
    )

    population = Population([genome])
    while True:
        population.evolve()
