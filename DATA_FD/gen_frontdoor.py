import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Union
import os


class DAGStructuredSCM:
    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 6),
        prior_hidden_size: Callable = lambda: np.random.randint(10, 50),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        edge_drop_prob: float = 0.4,
        activation: Callable = lambda x: np.tanh(x)
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.edge_drop_prob = edge_drop_prob
        self.activation = activation
        self.dag = None
        self.weights = {}
        self.biases = {}
        self.noise_distributions = {}
        self.feature_nodes = []
        self.node_values = {}
        self.topological_order = []
        
    def sample_noise_distribution(self) -> Callable:
        dist_type = np.random.choice([
            "normal", 
            "uniform", 
            "laplace", 
            "logistic"
        ])
        
        if dist_type == "normal":
            scale = np.random.uniform(0.1, 2.0)
            return lambda size: np.random.normal(0, scale, size)
        
        elif dist_type == "uniform":
            scale = np.random.uniform(0.1, 2.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        
        elif dist_type == "laplace":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.laplace(0, scale, size)
        
        elif dist_type == "logistic":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.logistic(0, scale, size)
    
    def construct_mlp_graph(self, num_layers: int, hidden_size: int) -> nx.DiGraph:
        G = nx.DiGraph()
        
        node_id = 0
        
        layer_sizes = [hidden_size] * num_layers
        
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        for layer_idx in range(num_layers - 1):
            src_nodes = nodes_by_layer[layer_idx]
            dst_nodes = nodes_by_layer[layer_idx + 1]
            
            for src in src_nodes:
                for dst in dst_nodes:
                    G.add_edge(src, dst)
        
        return G
    
    def transform_to_dag(self, G: nx.DiGraph) -> nx.DiGraph:
        dag = G.copy()
        
        edges = list(G.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)
        
        if num_edges_to_drop > 0:
            edges_to_drop = np.random.choice(
                len(edges), 
                size=num_edges_to_drop, 
                replace=False
            )
            
            for idx in edges_to_drop:
                u, v = edges[idx]
                dag.remove_edge(u, v)
        
        assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG after edge removal"
        
        return dag
    
    def sample_structural_equation_parameters(self) -> None:
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))
            
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
                
            self.biases[node] = self.prior_weight()
            
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def evaluate_node(self, node: int) -> float:
        parents = list(self.dag.predecessors(node))
        
        if not parents:
            noise = self.noise_distributions[node](1)[0]
            value = self.activation(self.biases[node] + noise)
        else:
            weighted_sum = sum(
                self.weights[node][parent] * self.node_values[parent] 
                for parent in parents
            )
            
            noise = self.noise_distributions[node](1)[0]
            value = self.activation(weighted_sum + self.biases[node] + noise)
            
        return value
    
    def sample_observation(self) -> np.ndarray:
        self.node_values = {}
        
        for node in self.topological_order:
            self.node_values[node] = self.evaluate_node(node)
        
        features = np.array([self.node_values[node] for node in self.feature_nodes])
        
        return features
    
    def generate_dataset(
        self, 
        num_features: int, 
        num_samples: int
    ) -> np.ndarray:
        num_layers = self.prior_layers()
        hidden_size = self.prior_hidden_size()
        
        mlp_graph = self.construct_mlp_graph(num_layers, hidden_size)
        
        self.dag = self.transform_to_dag(mlp_graph)
        
        self.topological_order = list(nx.topological_sort(self.dag))
        
        self.sample_structural_equation_parameters()
        
        all_nodes = list(self.dag.nodes())
        assert num_features <= len(all_nodes), "Requested more features than available nodes"
        self.feature_nodes = np.random.choice(
            all_nodes, 
            size=num_features, 
            replace=False
        )
        
        dataset = np.zeros((num_samples, num_features))
        for i in range(num_samples):
            dataset[i] = self.sample_observation()
            
        return dataset


class TreatmentAssignerWithConfounding:
    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x)
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.treatment_network = None
        self.weights = {}
        self.biases = {}
        self.output_node = None

    def construct_treatment_network(
        self, 
        num_layers: int, 
        hidden_size: int, 
        input_size: int
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)

        self.output_node = nodes_by_layer[-1][0]

        for layer_idx in range(num_layers - 1):
            for src in nodes_by_layer[layer_idx]:
                for dst in nodes_by_layer[layer_idx + 1]:
                    G.add_edge(src, dst)
        return G

    def sample_treatment_network_parameters(self) -> None:
        for node in self.treatment_network.nodes():
            parents = list(self.treatment_network.predecessors(node))
            if parents:
                self.weights[node] = {parent: self.prior_weight() for parent in parents}
            self.biases[node] = self.prior_weight()

    def forward_propagate(self, xu: np.ndarray) -> Dict[int, float]:
        node_values = {}
        node_layers = nx.get_node_attributes(self.treatment_network, 'layer')
        max_layer = max(node_layers.values())

        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            for node in layer_nodes:
                parents = list(self.treatment_network.predecessors(node))
                if layer == 0:
                    node_values[node] = xu[node] if node < len(xu) else 0.0
                else:
                    weighted_sum = sum(self.weights[node][p] * node_values[p] for p in parents)
                    value = weighted_sum + self.biases[node]
                    if node != self.output_node:
                        value = self.activation(value)
                    node_values[node] = value
        return node_values

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def generate_treatments(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_features_total = X.shape[1] + U.shape[1]

        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()

        self.treatment_network = self.construct_treatment_network(
            num_layers=num_layers,
            hidden_size=hidden_size,
            input_size=n_features_total
        )
        self.sample_treatment_network_parameters()

        treatments = np.zeros(n_samples, dtype=int)
        propensity_scores = np.zeros(n_samples, dtype=float)

        for i in range(n_samples):
            xu = np.concatenate([X[i], U[i]])
            node_values = self.forward_propagate(xu)
            logit = node_values[self.output_node]
            propensity = self.sigmoid(logit)
            propensity_scores[i] = propensity
            treatments[i] = np.random.binomial(1, propensity)

        self.propensity_scores = propensity_scores
        return treatments

class MediatorGenerator:
    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x)
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.mediator_network = None
        self.weights = {}
        self.biases = {}
        self.noise_distributions = {}
        self.output_node = None
        
    def sample_noise_distribution(self) -> Callable:
        dist_type = np.random.choice([
            "normal", 
            "uniform", 
            "laplace", 
            "logistic"
        ])
        
        if dist_type == "normal":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.normal(0, scale, size)
        elif dist_type == "uniform":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        elif dist_type == "laplace":
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.laplace(0, scale, size)
        elif dist_type == "logistic":
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.logistic(0, scale, size)
    
    def construct_mediator_network(
        self, 
        num_layers: int, 
        hidden_size: int, 
        input_size: int
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        
        node_id = 0
        
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]
        
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        self.output_node = nodes_by_layer[-1][0]
        
        for layer_idx in range(num_layers - 1):
            src_nodes = nodes_by_layer[layer_idx]
            dst_nodes = nodes_by_layer[layer_idx + 1]
            
            for src in src_nodes:
                for dst in dst_nodes:
                    G.add_edge(src, dst)
        
        return G
    
    def sample_mediator_network_parameters(self) -> None:
        for node in self.mediator_network.nodes():
            parents = list(self.mediator_network.predecessors(node))
            
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
                
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def forward_propagate(
        self, 
        xa: np.ndarray, 
        noise_samples: Dict[int, float]
    ) -> Dict[int, float]:
        node_values = {}
        
        node_layers = nx.get_node_attributes(self.mediator_network, 'layer')
        max_layer = max(node_layers.values())
        
        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            
            for node in layer_nodes:
                parents = list(self.mediator_network.predecessors(node))
                
                if layer == 0:
                    node_values[node] = xa[node] if node < len(xa) else 0
                else:
                    weighted_sum = sum(
                        self.weights[node][parent] * node_values[parent] 
                        for parent in parents
                    )
                    
                    noise = noise_samples[node]
                    value = weighted_sum + self.biases[node] + noise
                    
                    if node != self.output_node:
                        value = self.activation(value)
                    
                    node_values[node] = value
        
        return node_values
    
    def generate_mediators(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_features_total = X.shape[1] + 1
        
        num_layers = self.prior_layers()
        hidden_size = self.prior_hidden_size()
        num_layers = max(3, num_layers)
        
        self.mediator_network = self.construct_mediator_network(
            num_layers, 
            hidden_size, 
            n_features_total
        )
        
        self.sample_mediator_network_parameters()
        
        mediators = np.zeros(n_samples)
        
        for i in range(n_samples):
            xa = np.concatenate([X[i], [A[i]]])
            
            noise_samples = {
                node: self.noise_distributions[node](1)[0]
                for node in self.mediator_network.nodes()
            }
            
            node_values = self.forward_propagate(xa, noise_samples)
            
            mediators[i] = node_values[self.output_node]
        
        return mediators


class OutcomeGeneratorWithMediator:
    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        edge_drop_prob: float = 0.3,
        activation: Callable = lambda x: np.tanh(x)
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.edge_drop_prob = edge_drop_prob
        self.activation = activation
        self.outcome_network = None
        self.weights = {}
        self.biases = {}
        self.noise_distributions = {}
        self.output_node = None
        
    def sample_noise_distribution(self) -> Callable:
        dist_type = np.random.choice([
            "normal", 
            "uniform", 
            "laplace", 
            "logistic"
        ])
        
        if dist_type == "normal":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.normal(0, scale, size)
        elif dist_type == "uniform":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        elif dist_type == "laplace":
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.laplace(0, scale, size)
        elif dist_type == "logistic":
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.logistic(0, scale, size)
    
    def construct_outcome_network(
        self, 
        num_layers: int, 
        hidden_size: int, 
        input_size: int
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        
        node_id = 0
        
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]
        
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        self.output_node = nodes_by_layer[-1][0]
        
        for layer_idx in range(num_layers - 1):
            src_nodes = nodes_by_layer[layer_idx]
            dst_nodes = nodes_by_layer[layer_idx + 1]
            
            for src in src_nodes:
                for dst in dst_nodes:
                    G.add_edge(src, dst)
        
        edges = list(G.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)
        
        if num_edges_to_drop > 0:
            edges_to_drop = np.random.choice(
                len(edges), 
                size=num_edges_to_drop, 
                replace=False
            )
            
            for idx in edges_to_drop:
                u, v = edges[idx]
                if v != self.output_node:
                    G.remove_edge(u, v)
        
        return G
    
    def sample_outcome_network_parameters(self) -> None:
        for node in self.outcome_network.nodes():
            parents = list(self.outcome_network.predecessors(node))
            
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
                
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def forward_propagate(
        self, 
        xum: np.ndarray, 
        noise_samples: Dict[int, float]
    ) -> Dict[int, float]:
        node_values = {}
        
        node_layers = nx.get_node_attributes(self.outcome_network, 'layer')
        max_layer = max(node_layers.values())
        
        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            
            for node in layer_nodes:
                parents = list(self.outcome_network.predecessors(node))
                
                if layer == 0:
                    node_values[node] = xum[node] if node < len(xum) else 0
                else:
                    weighted_sum = sum(
                        self.weights[node][parent] * node_values[parent] 
                        for parent in parents
                    ) if parents else 0
                    
                    noise = noise_samples[node]
                    value = weighted_sum + self.biases[node] + noise
                    
                    if node != self.output_node:
                        value = self.activation(value)
                    
                    node_values[node] = value
        
        return node_values
    
    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))
    
    def generate_outcomes(
        self, 
        X: np.ndarray, 
        U: np.ndarray,
        M: np.ndarray,
        A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        n_features_total = X.shape[1] + U.shape[1] + 1
        
        num_layers = self.prior_layers()
        hidden_size = self.prior_hidden_size()
        num_layers = max(3, num_layers)
        
        self.outcome_network = self.construct_outcome_network(
            num_layers, 
            hidden_size, 
            n_features_total
        )
        
        self.sample_outcome_network_parameters()
        
        mediator_gen = MediatorGenerator(
            prior_layers=lambda: np.random.randint(2, 5),
            prior_hidden_size=lambda: np.random.randint(8, 30),
            prior_weight=lambda: np.random.normal(0, 1),
            activation=lambda x: np.tanh(x)
        )
        
        A0 = np.zeros(n_samples, dtype=int)
        A1 = np.ones(n_samples, dtype=int)
        M0 = mediator_gen.generate_mediators(X, A0)
        M1 = mediator_gen.generate_mediators(X, A1)
        
        Y = np.zeros(n_samples, dtype=float)
        Y0 = np.zeros(n_samples, dtype=float)
        Y1 = np.zeros(n_samples, dtype=float)
        
        self.Y0_probs = np.zeros(n_samples)
        self.Y1_probs = np.zeros(n_samples)
        
        for i in range(n_samples):
            noise_samples = {
                node: self.noise_distributions[node](1)[0]
                for node in self.outcome_network.nodes()
            }
            
            xum = np.concatenate([X[i], U[i], [M[i]]])
            node_values = self.forward_propagate(xum, noise_samples)
            Y[i] = node_values[self.output_node]
            
            xum0 = np.concatenate([X[i], U[i], [M0[i]]])
            node_values0 = self.forward_propagate(xum0, noise_samples)
            Y0[i] = node_values0[self.output_node]
            prob0 = self.sigmoid(node_values0[self.output_node])
            self.Y0_probs[i] = prob0
            
            xum1 = np.concatenate([X[i], U[i], [M1[i]]])
            node_values1 = self.forward_propagate(xum1, noise_samples)
            Y1[i] = node_values1[self.output_node]
            prob1 = self.sigmoid(node_values1[self.output_node])
            self.Y1_probs[i] = prob1
        
        return Y, Y0, Y1, M0, M1


def save_dataset_frontdoor(X: np.ndarray, U: np.ndarray, A: np.ndarray, 
                          M: np.ndarray, Y: np.ndarray,
                          Y0: np.ndarray = None, Y1: np.ndarray = None,
                          M0: np.ndarray = None, M1: np.ndarray = None,
                          filename: str = "synthetic_frontdoor_dataset.csv") -> pd.DataFrame:
    n_features_x = X.shape[1]
    n_features_u = U.shape[1]
    
    feature_names_x = [f"x{i}" for i in range(n_features_x)]
    feature_names_u = [f"u{i}" for i in range(n_features_u)]
    
    df = pd.DataFrame(X, columns=feature_names_x)
    df_u = pd.DataFrame(U, columns=feature_names_u)
    df = pd.concat([df, df_u], axis=1)
    
    df['treatment'] = A
    df['mediator'] = M
    df['outcome'] = Y
    
    if Y0 is not None:
        df['y0'] = Y0
    if Y1 is not None:
        df['y1'] = Y1
    if M0 is not None:
        df['m0'] = M0
    if M1 is not None:
        df['m1'] = M1
    
    if Y0 is not None and Y1 is not None:
        df['ite'] = Y1 - Y0
        df['ate'] = np.mean(Y1 - Y0)
    
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    return df


def generate_single_frontdoor_dataset(
    dataset_id: int,
    num_samples: int = 1024,
    num_features_x: int = 10,
    num_features_u: Optional[int] = None,
    output_dir: str = "data/frontdoor",
    seed_offset: int = 0,
    u_dim_sampler: Optional[Callable[[], int]] = None
):
    np.random.seed(dataset_id + seed_offset)

    print(f"\n=== Generating Front-Door Dataset {dataset_id} ===")

    if num_features_u is None:
        if u_dim_sampler is not None:
            num_features_u = int(u_dim_sampler())
        else:
            num_features_u = int(np.random.randint(1, 16))
    print(f"Using U dimension: {num_features_u}")

    
    dag_scm_x = DAGStructuredSCM(
        prior_layers=lambda: np.random.randint(3, 7),
        prior_hidden_size=lambda: np.random.randint(15, 40),
        prior_weight=lambda: np.random.normal(0, 1),
        edge_drop_prob=0.5,
        activation=lambda x: np.tanh(x)
    )
    X = dag_scm_x.generate_dataset(num_features_x, num_samples)
    print("Generated X (observed confounders)")
    
    dag_scm_u = DAGStructuredSCM(
        prior_layers=lambda: np.random.randint(3, 6),
        prior_hidden_size=lambda: np.random.randint(10, 30),
        prior_weight=lambda: np.random.normal(0, 1),
        edge_drop_prob=0.4,
        activation=lambda x: np.tanh(x)
    )
    U = dag_scm_u.generate_dataset(num_features_u, num_samples)
    print("Generated U (unobserved confounders)")
    
    treatment_assigner = TreatmentAssignerWithConfounding(
        prior_layers=lambda: np.random.randint(3, 5),
        prior_hidden_size=lambda: np.random.randint(8, 20),
        prior_weight=lambda: np.random.normal(0, 0.8),
        activation=lambda x: np.tanh(x)
    )
    A = treatment_assigner.generate_treatments(X, U)
    print("Generated treatment A")
    
    mediator_generator = MediatorGenerator(
        prior_layers=lambda: np.random.randint(3, 5),
        prior_hidden_size=lambda: np.random.randint(10, 25),
        prior_weight=lambda: np.random.normal(0, 1.0),
        activation=lambda x: np.tanh(x)
    )
    M = mediator_generator.generate_mediators(X, A)
    print("Generated mediator M")
    
    outcome_generator = OutcomeGeneratorWithMediator(
        prior_layers=lambda: np.random.randint(3, 6),
        prior_hidden_size=lambda: np.random.randint(10, 25),
        prior_weight=lambda: np.random.normal(0, 1.0),
        edge_drop_prob=0.4,
        activation=lambda x: np.tanh(x)
    )
    Y, Y0, Y1, M0, M1 = outcome_generator.generate_outcomes(X, U, M, A)
    
    print("Generated outcomes Y")

    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"synthetic_frontdoor_dataset_{dataset_id}.csv")
    dataset = save_dataset_frontdoor(X, U, A, M, Y, Y0, Y1, M0, M1, filename)
    
    return dataset, X, U, A, M, Y, Y0, Y1


def generate_multiple_frontdoor_datasets(num_datasets: int = 10, num_samples: int = 1024,
                                        num_features_x: int = 10, num_features_u: int = 5,
                                        output_dir: str = "data/frontdoor",
                                        base_seed: int = 42):
    print(f"Generating {num_datasets} front-door datasets with {num_samples} samples each...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, num_datasets + 1):
        try:
            dataset, X, U, A, M, Y, Y0, Y1 = generate_single_frontdoor_dataset(
                dataset_id=i,
                num_samples=num_samples,
                num_features_x=num_features_x,
                num_features_u=num_features_u,
                output_dir=output_dir,
                seed_offset=base_seed
            )
            
        except Exception as e:
            print(f"Error generating dataset {i}: {str(e)}")
            continue
    
    print(f"\n=== Generation Complete ===")
    print(f"Generated {num_datasets} datasets successfully")
    
    return None


if __name__ == "__main__":
    generate_multiple_frontdoor_datasets(
        num_datasets=10,
        num_samples=1024,
        num_features_x=10,
        num_features_u= None,
        output_dir="DATA_FD/frontdoor",
        base_seed=0
    )