import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import random

from meat.ga import *

random.seed(42)

def create_sequences(features, labels, price_changes, window_size=10):
    feature_sequences = []
    label_sequences = []
    price_change_sequences = []
    for i in range(len(features) - window_size):
        feature_sequences.append(features[i:i + window_size])
        label_one_hot = [1, 0] if labels[i + window_size] == 1 else [0, 1]
        label_sequences.append(label_one_hot)
        price_change_sequences.append(price_changes[i + window_size])
    return np.array(feature_sequences), np.array(label_sequences), np.array(price_change_sequences)
    
class StockDataset(Dataset):
    def __init__(self, features, labels, price_changes):
        self.features = features
        self.labels = labels
        self.price_changes = price_changes  # Add this

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.price_changes[idx], dtype=torch.float))  # Add this

class Simulator:
    def __init__(self, gene_pool: list[Gene], capacity: int = 16):
        self.gene_pool = gene_pool

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.generation = 0
        self.epochs = 20

        # Backtesting parameters
        self.trade_thershold = 0.55
        self.long_only = False

        self._initialize()

    def _initialize(self):
        # Load the data
        file_path = './data/augmented/AAPL.csv'
        data = pd.read_csv(file_path)

        # Display the first few rows of the dataframe and some additional info to understand its structure
        data_head = data.head()
        data_description = data.describe()
        data_info = data.info()

        data_head, data_description, data_info

        # Fill missing values
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)  # Backfill if the first row contains NaN

        # Select features and labels if you have specific output in mind
        features = data[[
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Log_Open', 'Log_Open', 'Log_Open', 'Log_Open', 'Log_Open', 
            'RSI_7', 'RSI_14', 'RSI_21', 
            'EMA_9', 'EMA_21', 'EMA_55'
        ]]
        labels = data['Direction']  # Example: Predicting next day's return
        price_changes = data['Price_Change'].values

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Create sequenced features and labels
        feature_sequences, label_sequences, price_change_sequences = create_sequences(features_scaled, labels, price_changes, window_size=10)

        # Split the data into training and testing sets
        #X_train, X_test, y_train, y_test, pc_train, pc_test = train_test_split(feature_sequences, label_sequences, price_change_sequences, test_size=0.2, random_state=42)

        X_train_val, X_test, y_train_val, y_test, pc_train_val, pc_test = train_test_split(feature_sequences, label_sequences, price_change_sequences, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val, pc_train, pc_val = train_test_split(X_train_val, y_train_val, pc_train_val, test_size=0.25, random_state=42)

        # Create instances of the StockDataset
        train_dataset = StockDataset(X_train, y_train, pc_train)
        val_dataset = StockDataset(X_val, y_val, pc_val)

        # Create DataLoaders for training and testing
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(StockDataset(X_test, y_test, pc_test), batch_size=64, shuffle=False)

    def train_loop(self, dataloader, model, loss_fn, optimizer):
        total_loss = 0
        #size = len(dataloader.dataset)
        model.train()
        for batch, (X, y, pc) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            X = X.permute(0, 2, 1)
            y = y.float()

            pred = nn.Softmax(dim=1)(model(X))
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return total_loss / len(dataloader)

    def validate(self, model: nn.Module, dataloader, loss_fn):
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y, pc in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                X = X.permute(0, 2, 1)
                y = y.float()

                pred = nn.Softmax(dim=1)(model(X))
                loss = loss_fn(pred, y)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def backtest(self, model, dataloader):
        model.eval()
        pnl_list = [1]  # Start with an initial PnL of 1
        with torch.no_grad():
            for X, _, price_change in dataloader:
                X = X.to(self.device).permute(0, 2, 1)
                preds = nn.Softmax(dim=1)(model(X))

                positions = torch.zeros(preds.shape[0]).to(self.device)
                positions[preds[:, 0] > self.trade_thershold] = 1
                positions[preds[:, 1] > self.trade_thershold] = 0 if self.long_only else -1

                # Calculate daily returns based on positions and price changes
                daily_returns: torch.Tensor = positions * price_change.to(self.device)
                #print(1 + daily_returns)
                pnl = (1 + daily_returns).cumprod(dim=0)
                pnl_list.extend(pnl.flatten().tolist())  # Append all elements of the cumulative product

        return pnl_list

    def sharpe_ratio(self, pnl):
        daily_returns = np.diff(pnl) / pnl[:-1]
        mean_daily_returns = np.mean(daily_returns)
        std_daily_returns = np.std(daily_returns)
        sharpe_ratio = mean_daily_returns / std_daily_returns
        return sharpe_ratio

    def sortino_ratio(self, pnl):
        daily_returns = np.diff(pnl) / pnl[:-1]
        mean_daily_returns = np.mean(daily_returns)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
        risk_free_rate = 0
        sortino_ratio = (mean_daily_returns - risk_free_rate) / downside_deviation
        return sortino_ratio
    
    def max_drawdown(self, pnl):
        running_max = np.maximum.accumulate(pnl)
        drawdowns = (running_max - pnl) / running_max
        max_drawdown = np.max(drawdowns)
        return max_drawdown

    def _evaluate(self, gene) -> tuple[Gene, float]:
        model = Model(gene).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            train_loss = self.train_loop(self.train_loader, model, loss_fn, optimizer)
            val_loss = self.validate(model, self.val_loader, loss_fn)
            train_backtest = self.backtest(model, self.train_loader)
            val_backtest = self.backtest(model, self.val_loader)
            test_backtest = self.backtest(model, self.test_loader)
            sharpe_ratio = self.sharpe_ratio(test_backtest)
            sortino_ratio = self.sortino_ratio(test_backtest)
            max_drawdown = self.max_drawdown(test_backtest)

        return {
            "gene": gene, 
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            'train_backtest': train_backtest,
            "val_backtest": val_backtest, 
            "test_backtest": test_backtest,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown
        }

    def evaluate(self):
        results = []
        with ThreadPoolExecutor() as executor:
            results = [executor.submit(self._evaluate, gene) for gene in self.gene_pool]
        self.generation_results = [result.result() for result in results]

        """for gene in self.gene_pool:
            try:
                results.append(self._evaluate(gene))
            except Exception as e:
                print(f"Error: {e}")
                print(f"Nodes: {gene.nodes}")
                print(f"Edges: {gene.edges}")
                print()
        #results = [self._evaluate(gene) for gene in self.gene_pool]
        self.generation_losses = results"""

    def select(self):
        #print(f"Generation {self.generation} losses: {self.generation_losses}")
        self.generation_results.sort(key=lambda x: x['train_backtest'][-1], reverse=True)
        self.gene_pool = [result['gene'] for result in self.generation_results[:8]]

    def evolve(self):
        print()
        print(f"================ Generation: {self.generation:2} ================")
        print(f"Old Population: {len(self.gene_pool)}\n")
        self.evaluate()
        self.select()
        print("Train loss:", [f'{result["train_loss"]:.6f}' for result in self.generation_results])
        print("Val loss:", [f'{result["val_loss"]:.6f}' for result in self.generation_results])
        print("Train backtest result:", [f'{result["train_backtest"][-1]:.4f}' for result in self.generation_results])
        print("Val backtest result:", [f'{result["val_backtest"][-1]:.4f}' for result in self.generation_results])
        print("Test backtest result:", [f'{result["test_backtest"][-1]:.4f}' for result in self.generation_results])
        print("Sharpe ratio:", [f'{result["sharpe_ratio"]:.4f}' for result in self.generation_results])
        print("Sortino ratio:", [f'{result["sortino_ratio"]:.4f}' for result in self.generation_results])
        print("Max drawdown:", [f'{result["max_drawdown"]:.4f}' for result in self.generation_results])
        #for i, (gene, train_loss, val_loss) in enumerate(self.generation_losses):
        #    print(f"Model {i}: Train loss {train_loss:.4f} Val loss {val_loss} Nodes {len(gene.nodes)} Edges {gene.edges}")

        print(f"\n==== Overview ====")
        print(f"Generation train backtest result: {sum([result['train_backtest'][-1] for result in self.generation_results])/len(self.generation_results):.6f}")
        print(f"Generation val backtest result: {sum([result['val_backtest'][-1] for result in self.generation_results])/len(self.generation_results):.6f}")
        print(f"Generation test backtest result: {sum([result['test_backtest'][-1] for result in self.generation_results])/len(self.generation_results):.6f}")
        print(f"Generation sharpe ratio: {sum([result['sharpe_ratio'] for result in self.generation_results])/len(self.generation_results):.6f}")
        print(f"Generation sortino ratio: {sum([result['sortino_ratio'] for result in self.generation_results])/len(self.generation_results):.6f}")
        print(f"Generation max drawdown: {sum([result['max_drawdown'] for result in self.generation_results])/len(self.generation_results):.6f}")

        new_gene_pool = []
        for gene in self.gene_pool:
            new_gene_pool.append(gene.mutate())
            new_gene_pool.append(gene.mutate())
            new_gene_pool.append(gene.mutate())
        self.gene_pool.extend(new_gene_pool)

        self.generation += 1
        print(f"\nNew Population: {len(self.gene_pool)}")
        print()
        print(f"Best model:")
        #print(f"Nodes {self.generation_losses[0][0].nodes}")
        #print(f"Edges {self.generation_losses[0][0].edges}")
        print(f"Train backtest result: {self.generation_results[0]['train_backtest'][-1]}")
        print(f"Val backtest result: {self.generation_results[0]['val_backtest'][-1]}")
        print(f"Test backtest result: {self.generation_results[0]['test_backtest'][-1]}")
        print(f"Sharpe ratio: {self.generation_results[0]['sharpe_ratio']}")
        print(f"Sortino ratio: {self.generation_results[0]['sortino_ratio']}")
        print(f"Max drawdown: {self.generation_results[0]['max_drawdown']}")
        print("================================================")

if __name__ == "__main__":
    """genome = Gene(
        nodes={
            'in': Node(in_shape=None, out_shape=(11,10), out_spatial=True),
            '0': Node(nn.Flatten(), in_shape=(11,10), out_shape=(110,), out_spatial=False),
            '1': Node(nn.Linear(110, 1), nn.ReLU(), in_shape=(110,), out_shape=(1,), out_spatial=False),
            'out': Node(in_shape=(1,), out_shape=None, out_spatial=False)
        }, edges = [
            ('in', '0'),
            ('0', '1'),
            ('1', 'out'),
        ]
    )"""

    genome = Gene(
        nodes={
            'in': Node(in_shape=None, out_shape=(16,10), out_spatial=True),
            '0': Node(SelfAttention(16, 4), in_shape=(16,10), out_shape=(16,10), out_spatial=True),
            '1': Node(nn.Flatten(), in_shape=(16,10), out_shape=(160,), out_spatial=False),
            '2': Node(nn.Linear(160, 2), nn.ReLU(), in_shape=(160,), out_shape=(2,), out_spatial=False),
            'out': Node(in_shape=(2,), out_shape=None, out_spatial=False)
        }, edges = [
            ('in', '0'),
            ('0', '1'),
            ('1', '2'),
            ('2', 'out'),
        ]
    )

    genome = Gene(
        nodes = {
            'in': Node(in_shape=None, out_shape=(16,10), out_spatial=True),
            '0': Node(nn.Flatten(), in_shape=(16,10), out_shape=(160,), out_spatial=False),
            '1': Node(nn.Linear(160, 2), nn.ReLU(), in_shape=(160,), out_shape=(2,), out_spatial=False),
            'out': Node(in_shape=(2,), out_shape=None, out_spatial=False)
        }, edges = [
            ('in', '0'),
            ('0', '1'),
            ('1', 'out'),
        ]
    )

    mutation = genome._mutate_add_two_node(in_node='in', out_node='0')
    print(mutation.nodes)
    print(mutation.edges)

    simulator = Simulator([genome])

    for i in range(10):
        simulator.evolve()
