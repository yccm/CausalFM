import numpy as np
import networkx as nx
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Union
import os


class DAGStructuredSCM:
    """
    Implementation of DAG-structured Structural Causal Models (SCMs) sampler
    based on a modified MLP architecture.
    """
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
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
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
            for src in nodes_by_layer[layer_idx]:
                for dst in nodes_by_layer[layer_idx + 1]:
                    G.add_edge(src, dst)
        return G
    
    def transform_to_dag(self, G: nx.DiGraph) -> nx.DiGraph:
        dag = G.copy()
        edges = list(G.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)
        if num_edges_to_drop > 0:
            edges_to_drop_indices = np.random.choice(len(edges), size=num_edges_to_drop, replace=False)
            edges_to_drop = [edges[i] for i in edges_to_drop_indices]
            dag.remove_edges_from(edges_to_drop)
        assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
        return dag
    
    def sample_structural_equation_parameters(self) -> None:
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))
            if parents:
                self.weights[node] = {parent: self.prior_weight() for parent in parents}
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def evaluate_node(self, node: int) -> float:
        parents = list(self.dag.predecessors(node))
        noise = self.noise_distributions[node](1)[0]
        if not parents:
            value = self.activation(self.biases[node] + noise)
        else:
            weighted_sum = sum(self.weights[node][parent] * self.node_values[parent] for parent in parents)
            value = self.activation(weighted_sum + self.biases[node] + noise)
        return value
    
    def sample_observation(self) -> np.ndarray:
        self.node_values = {}
        for node in self.topological_order:
            self.node_values[node] = self.evaluate_node(node)
        return np.array([self.node_values[node] for node in self.feature_nodes])
    
    def generate_dataset(self, num_features: int, num_samples: int) -> np.ndarray:
        num_layers = self.prior_layers()
        hidden_size = self.prior_hidden_size()
        mlp_graph = self.construct_mlp_graph(num_layers, hidden_size)
        self.dag = self.transform_to_dag(mlp_graph)
        self.topological_order = list(nx.topological_sort(self.dag))
        self.sample_structural_equation_parameters()
        all_nodes = list(self.dag.nodes())
        assert num_features <= len(all_nodes), "More features requested than available nodes"
        self.feature_nodes = np.random.choice(all_nodes, size=num_features, replace=False)
        dataset = np.array([self.sample_observation() for _ in range(num_samples)])
        return dataset


class BaseMLPGenerator:
    """A base class for MLP-like network generators."""
    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x),
        use_layer_noise: bool = True,
        layer_noise_scale: float = 0.1,  # Fixed: Added noise scale parameter
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.use_layer_noise = use_layer_noise
        self.layer_noise_scale = layer_noise_scale  # Fixed: Store noise scale

        self.network = None
        self.weights = {}
        self.biases = {}
        self.noise_distributions = {}
        self.output_node = None

    def sample_noise_distribution(self) -> Callable:
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        scale_map = {"normal": (0.1, 1.0), "uniform": (0.1, 1.0), "laplace": (0.1, 0.5), "logistic": (0.1, 0.5)}
        scale = np.random.uniform(*scale_map[dist_type])
        dist_map = {
            "normal": lambda s: np.random.normal(0, scale, s),
            "uniform": lambda s: np.random.uniform(-scale, scale, s),
            "laplace": lambda s: np.random.laplace(0, scale, s),
            "logistic": lambda s: np.random.logistic(0, scale, s)
        }
        return dist_map[dist_type]

    def _construct_network(self, num_layers: int, hidden_size: int, input_size: int, output_size: int) -> nx.DiGraph:
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [output_size]
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = [node_id + i for i in range(size)]
            nodes_by_layer.append(layer_nodes)
            for node in layer_nodes:
                G.add_node(node, layer=layer_idx)
            node_id += size

        if output_size == 1:
            self.output_node = nodes_by_layer[-1][0]
        
        for i in range(num_layers - 1):
            for src in nodes_by_layer[i]:
                for dst in nodes_by_layer[i+1]:
                    G.add_edge(src, dst)
        return G

    def _sample_network_parameters(self) -> None:
        for node in self.network.nodes():
            parents = list(self.network.predecessors(node))
            if parents:
                self.weights[node] = {parent: self.prior_weight() for parent in parents}
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()

    def _forward_propagate(self, z: np.ndarray) -> float:
        node_values = {}
        node_layers = nx.get_node_attributes(self.network, 'layer')
        
        for i in range(len(z)):
            node_values[i] = z[i]
            
        for layer in sorted(list(set(node_layers.values())))[1:]:
            for node in [n for n, l in node_layers.items() if l == layer]:
                parents = list(self.network.predecessors(node))
                weighted_sum = sum(self.weights[node][p] * node_values[p] for p in parents)

                if self.use_layer_noise:
                    noise = np.random.normal(0, self.layer_noise_scale)
                else:
                    noise = 0.0
                
                value = weighted_sum + self.biases[node] + noise
                if node != self.output_node:
                    value = self.activation(value)
                node_values[node] = value
                
        return node_values[self.output_node]
    
    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))



class InstrumentGenerator(BaseMLPGenerator):
    """Generates an instrument Z from covariates X."""
    def generate_instrument(self, X: np.ndarray, instrument_type: str = 'binary') -> np.ndarray:
        n_samples, n_features = X.shape
        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()
        
        self.network = self._construct_network(num_layers, hidden_size, n_features, 1)
        self._sample_network_parameters()
        
        Z = np.zeros(n_samples)
        for i in range(n_samples):
            logit = self._forward_propagate(X[i])
            if instrument_type == 'binary':
                prob = self.sigmoid(logit)
                Z[i] = np.random.binomial(1, prob)
            else: 
                Z[i] = logit
        
        return Z


class IVTreatmentAssigner(BaseMLPGenerator):

    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x),
    ):
        
        super().__init__(
            prior_layers=prior_layers,
            prior_hidden_size=prior_hidden_size,
            prior_weight=prior_weight,
            activation=activation,
            use_layer_noise=False,   
        )

    def generate_treatments(self, X: np.ndarray, Z: np.ndarray, U: np.ndarray) -> np.ndarray:
        
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        full_input = np.hstack([X, Z, U])
        n_samples, n_features_full = full_input.shape
        
        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()
        
        self.network = self._construct_network(num_layers, hidden_size, n_features_full, 1)
        self._sample_network_parameters()
        
        A = np.zeros(n_samples, dtype=int)
        self.propensity_scores = np.zeros(n_samples)

        for i in range(n_samples):
            logit = self._forward_propagate(full_input[i])  
            propensity = self.sigmoid(logit)
            self.propensity_scores[i] = propensity
            A[i] = np.random.binomial(1, propensity)        
            
        return A

 

class IVOutcomeGenerator:

    def __init__(self,
        prior_layers: Callable = lambda: np.random.randint(3, 6),
        prior_hidden_size: Callable = lambda: np.random.randint(10, 25),
        prior_weight: Callable = lambda: np.random.normal(0, 1.0),
        activation: Callable = lambda x: np.tanh(x),
        outcome_noise_scale: float = 0.5):
        
        
        self.f_generator = BaseMLPGenerator(
            prior_layers, prior_hidden_size, prior_weight, activation,
            use_layer_noise=False
        )
        self.g_generator = BaseMLPGenerator(
            prior_layers, prior_hidden_size, prior_weight, activation,
            use_layer_noise=False
        )
        self.outcome_noise_scale = outcome_noise_scale

    def generate_outcomes(self, X: np.ndarray, A: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]

        
        input_f = np.hstack([X, A.reshape(-1, 1)])
        n_features_f = input_f.shape[1]
        num_layers_f = max(3, self.f_generator.prior_layers())
        hidden_size_f = self.f_generator.prior_hidden_size()
        self.f_generator.network = self.f_generator._construct_network(num_layers_f, hidden_size_f, n_features_f, 1)
        self.f_generator._sample_network_parameters()

        
        input_g = np.hstack([X, U])
        n_features_g = input_g.shape[1]
        num_layers_g = max(3, self.g_generator.prior_layers())
        hidden_size_g = self.g_generator.prior_hidden_size()
        self.g_generator.network = self.g_generator._construct_network(num_layers_g, hidden_size_g, n_features_g, 1)
        self.g_generator._sample_network_parameters()

        
        Y = np.zeros(n_samples)
        Y0 = np.zeros(n_samples)
        Y1 = np.zeros(n_samples)

        
        outcome_noise = np.random.normal(0, self.outcome_noise_scale, n_samples)

        for i in range(n_samples):
            
            g_val = self.g_generator._forward_propagate(input_g[i])

            
            f_val_factual = self.f_generator._forward_propagate(np.hstack([X[i], A[i]]))
            f_val_0 = self.f_generator._forward_propagate(np.hstack([X[i], 0]))
            f_val_1 = self.f_generator._forward_propagate(np.hstack([X[i], 1]))

            Y[i]  = f_val_factual + g_val + outcome_noise[i]
            Y0[i] = f_val_0       + g_val + outcome_noise[i]
            Y1[i] = f_val_1       + g_val + outcome_noise[i]
            
        return Y, Y0, Y1



def save_dataset_iv(df: pd.DataFrame, filename: str = "synthetic_iv_dataset.csv") -> None:
    """Saves the generated IV dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def generate_single_iv_dataset(dataset_id: int, num_samples: int = 1024, num_features: int = 10,
                               num_confounders_range: Tuple[int, int] = (2, 5),
                               output_dir: str = "data/iv_continuous", seed_offset: int = 0):
    """
    Generates a single synthetic dataset for the IV setting.
    """
    np.random.seed(dataset_id + seed_offset)
    print(f"\n=== Generating IV Dataset {dataset_id} ===")
    

    num_total_features = num_features + num_confounders_range[1] 
    scm = DAGStructuredSCM()
    W = scm.generate_dataset(num_total_features, num_samples)
    

    num_confounders = np.random.randint(num_confounders_range[0], num_confounders_range[1] + 1)
    U = W[:, :num_confounders]
    X = W[:, num_confounders:num_confounders + num_features]
    print(f"Data split: {X.shape[1]} observed features (X), {U.shape[1]} unobserved confounders (U)")


    instrument_gen = InstrumentGenerator()
    # instrument_type = np.random.choice(['binary', 'continuous'])
    instrument_type ='continuous'
    Z = instrument_gen.generate_instrument(X, instrument_type)
    print(f"Generated '{instrument_type}' instrument Z.")


    iv_treatment_assigner = IVTreatmentAssigner()
    A = iv_treatment_assigner.generate_treatments(X, Z, U)
    print("Generated treatment")


    iv_outcome_gen = IVOutcomeGenerator()
    Y, Y0, Y1 = iv_outcome_gen.generate_outcomes(X, A, U)
    print("Generated outcomes.")


    os.makedirs(output_dir, exist_ok=True)
    

    df_X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    df_U = pd.DataFrame(U, columns=[f'u{i}' for i in range(U.shape[1])])
    df_Z = pd.DataFrame(Z.reshape(-1, 1), columns=['z'])
    df_A = pd.DataFrame(A, columns=['treatment'])
    df_Y = pd.DataFrame({'outcome': Y, 'y0': Y0, 'y1': Y1, 'ite': Y1 - Y0})
    
    full_df = pd.concat([df_X, df_U, df_Z, df_A, df_Y], axis=1)
    
    filename = os.path.join(output_dir, f"synthetic_iv_dataset_{dataset_id}.csv")
    save_dataset_iv(full_df, filename)
    
    return full_df

def generate_multiple_iv_datasets(num_datasets: int = 10, num_samples: int = 1024,
                                 num_features: int = 10, output_dir: str = "data/iv_continuous",
                                 base_seed: int = 2024):
    """Generates multiple synthetic causal datasets for the IV setting."""
    print(f"Generating {num_datasets} IV datasets with {num_samples} samples each...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, num_datasets + 1):
        try:
            df = generate_single_iv_dataset(
                dataset_id=i,
                num_samples=num_samples,
                num_features=num_features,
                output_dir=output_dir,
                seed_offset=base_seed
            )
        except Exception as e:
            print(f"Error generating dataset {i}: {str(e)}")
            continue
            
    print("\n=== Generation Complete ===")
    return None

# Main execution block
if __name__ == "__main__":
    # Generate 10k datasets
    summary_stats = generate_multiple_iv_datasets(
        num_datasets=10,
        num_samples=1024,
        num_features=10,
        output_dir="DATA_IV/conti_Z_conti_Y",
        base_seed=42
    )