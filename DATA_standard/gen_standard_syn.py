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
        """
        Initialize the DAG-structured SCM sampler.
        
        Args:
            prior_layers: Distribution for number of layers
            prior_hidden_size: Distribution for hidden size
            prior_weight: Distribution for edge weights
            edge_drop_prob: Probability of dropping an edge to create DAG
            activation: Activation function to use (default: tanh)
        """
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
        """
        Sample a noise distribution from a meta-distribution.
        """
        # Sample a distribution type
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
        """
        Construct an MLP-like directed graph structure.
        
        Args:
            num_layers: Number of layers in the MLP
            hidden_size: Size of hidden layers
            
        Returns:
            A directed graph representing the MLP structure
        """
        G = nx.DiGraph()
        
        # Create nodes
        node_id = 0
        
        # Layer sizes 
        layer_sizes = [hidden_size] * num_layers
        
        # Create nodes 
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        # Create edges 
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
            # Get parents of the node
            parents = list(self.dag.predecessors(node))
            
            if parents:
                # Sample weights 
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
                
            # Sample bias term
            self.biases[node] = self.prior_weight()
            
            # Sample noise distribution
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
        
        # Process nodes in topological order
        for node in self.topological_order:
            self.node_values[node] = self.evaluate_node(node)
        
        # Extract feature values
        features = np.array([self.node_values[node] for node in self.feature_nodes])
        
        return features
    
    def generate_dataset(
        self, 
        num_features: int, 
        num_samples: int
    ) -> np.ndarray:

        # Sample number of layers and hidden size from priors
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
    


class OutcomeGenerator:
    """
    Implementation of the outcome generation algorithm based on an MLP network.
    """
    
    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        edge_drop_prob: float = 0.3,
        activation: Callable = lambda x: np.tanh(x)
    ):
        """
        Initialize the outcome generator.
        
        Args:
            prior_layers: Distribution for number of layers
            prior_hidden_size: Distribution for hidden size
            prior_weight: Distribution for edge weights
            edge_drop_prob: Probability of dropping edges to induce sparsity
            activation: Activation function to use (default: tanh)
        """
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

        # Sample a distribution type
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
        
        # Define layer sizes 
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]  # Output layer is size 1
        
        # Create nodes 
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        # Set output node
        self.output_node = nodes_by_layer[-1][0]
        
        # Create edges 
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
    
    def sigmoid(self, x: float) -> float:

        return 1.0 / (1.0 + np.exp(-x))
    
    def forward_propagate(
        self, 
        z: np.ndarray, 
        noise_samples: Dict[int, float]
    ) -> Dict[int, float]:

        node_values = {}
        
        # Get nodes 
        node_layers = nx.get_node_attributes(self.outcome_network, 'layer')
        max_layer = max(node_layers.values())
        
        # Process each layer in order
        for layer in range(max_layer + 1):
            # Get nodes in this layer
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            
            for node in layer_nodes:
                parents = list(self.outcome_network.predecessors(node))
                
                if layer == 0:
                    node_values[node] = z[node] 
                else:
                    if parents:
                        weighted_sum = sum(
                            self.weights[node][parent] * node_values[parent] 
                            for parent in parents
                        )
                    else:
                        weighted_sum = 0
                    
                    noise = noise_samples[node]
                    
                    value = weighted_sum + self.biases[node] + noise
                    if node != self.output_node:  
                        value = self.activation(value)
                    
                    node_values[node] = value
        
        return node_values
    
    def generate_outcomes(
        self, 
        X: np.ndarray, 
        A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_samples, n_features = X.shape
        
        # Sample number of layers and hidden size from priors
        num_layers = self.prior_layers()
        hidden_size = self.prior_hidden_size()
        
        # Ensure at least 3 layers (input, hidden, output)
        num_layers = max(3, num_layers)
        
        # Construct outcome network
        self.outcome_network = self.construct_outcome_network(
            num_layers, 
            hidden_size, 
            n_features + 1  # +1 for treatment
        )
        
        # Sample outcome network parameters
        self.sample_outcome_network_parameters()
        
        # Generate outcomes for each sample

        Y = np.zeros(n_samples, dtype=float)  # Factual continuous outcomes
        Y0 = np.zeros(n_samples, dtype=float)  # Potential continuous outcomes under A=0
        Y1 = np.zeros(n_samples, dtype=float)  # Potential continuous outcomes under A=1
        
        # Store probabilities for debugging and visualization
        self.Y0_probs = np.zeros(n_samples)
        self.Y1_probs = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Sample noise for all nodes - use the same noise for counterfactuals
            noise_samples = {
                node: self.noise_distributions[node](1)[0]
                for node in self.outcome_network.nodes()
            }
            
            # Create input vector for A=0
            z0 = np.concatenate([X[i], [0]])
            
            # Forward propagate through the network for A=0
            node_values0 = self.forward_propagate(z0, noise_samples)
            
            # Apply sigmoid to get probability
            prob0 = self.sigmoid(node_values0[self.output_node])
            self.Y0_probs[i] = prob0
            

            # Use continuous probability as outcome
            Y0[i] = node_values0[self.output_node]
            
            # Create input vector for A=1
            z1 = np.concatenate([X[i], [1]])
            
            # Forward propagate through the network for A=1
            node_values1 = self.forward_propagate(z1, noise_samples)
            
            # Apply sigmoid to get probability
            prob1 = self.sigmoid(node_values1[self.output_node])
            self.Y1_probs[i] = prob1
            
            Y1[i] = node_values1[self.output_node]
            
            # Set factual outcome based on actual treatment
            Y[i] = Y0[i] if A[i] == 0 else Y1[i]
        
        return Y, Y0, Y1
    
    


class TreatmentAssigner:
    """
    Implementation of the binary treatment assignment algorithm based on an MLP network.
    Takes covariates X and generates binary treatment assignments.
    """
    
    def __init__(
        self,
        prior_layers: Callable = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable = lambda: np.random.randint(8, 30),
        prior_weight: Callable = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x)
    ):
        """
        Initialize the treatment assignment generator.
        
        Args:
            prior_layers: Distribution for number of layers
            prior_hidden_size: Distribution for hidden size
            prior_weight: Distribution for edge weights
            activation: Activation function to use (default: tanh)
        """
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.treatment_network = None
        self.weights = {}
        self.biases = {}
        self.noise_distributions = {}
        self.output_node = None
        
    def sample_noise_distribution(self) -> Callable:
        """
        Sample a noise distribution from a meta-distribution.
        
        Returns:
            A callable that samples from the chosen noise distribution
        """
        # Sample a distribution type
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
    
    def construct_treatment_network(
        self, 
        num_layers: int, 
        hidden_size: int, 
        input_size: int
    ) -> nx.DiGraph:
        """
        Construct an MLP-like network for treatment assignment.
        
        Args:
            num_layers: Number of layers in the MLP
            hidden_size: Size of hidden layers
            input_size: Number of input features
            
        Returns:
            A directed graph representing the MLP structure
        """
        G = nx.DiGraph()
        
        # Create nodes
        node_id = 0
        
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]  
        
        # Create nodes for each layer
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        # Set output node
        self.output_node = nodes_by_layer[-1][0]
        
        for layer_idx in range(num_layers - 1):
            src_nodes = nodes_by_layer[layer_idx]
            dst_nodes = nodes_by_layer[layer_idx + 1]
            
            for src in src_nodes:
                for dst in dst_nodes:
                    G.add_edge(src, dst)
        
        return G
    
    def sample_treatment_network_parameters(self) -> None:
        """
        Sample weights, biases, and noise distributions for each node in the treatment network.
        """
        for node in self.treatment_network.nodes():
            parents = list(self.treatment_network.predecessors(node))
            
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
                
            self.biases[node] = self.prior_weight()
            
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def forward_propagate(
        self, 
        x: np.ndarray, 
        noise_samples: Dict[int, float]
    ) -> Dict[int, float]:

        node_values = {}
        
        node_layers = nx.get_node_attributes(self.treatment_network, 'layer')
        max_layer = max(node_layers.values())
        
        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            
            for node in layer_nodes:
                parents = list(self.treatment_network.predecessors(node))
                

                if layer == 0:
                    node_values[node] = x[node]  
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
    
    def sigmoid(self, x: float) -> float:

        return 1.0 / (1.0 + np.exp(-x))
    
    def generate_treatments(self, X: np.ndarray) -> np.ndarray:

        n_samples, n_features = X.shape
        
        # Sample number of layers and hidden size from priors
        num_layers = self.prior_layers()
        hidden_size = self.prior_hidden_size()
        
        num_layers = max(3, num_layers)
        
        # Construct treatment network
        self.treatment_network = self.construct_treatment_network(
            num_layers, 
            hidden_size, 
            n_features
        )
        
        self.sample_treatment_network_parameters()
        
        # Generate treatments for each sample
        treatments = np.zeros(n_samples, dtype=int)
        propensity_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            noise_samples = {
                node: self.noise_distributions[node](1)[0]
                for node in self.treatment_network.nodes()
            }
            
            node_values = self.forward_propagate(X[i], noise_samples)
            
            propensity = self.sigmoid(node_values[self.output_node])
            propensity_scores[i] = propensity
            
            # Sample treatment from Bernoulli distribution
            treatments[i] = np.random.binomial(1, propensity)
        
        self.propensity_scores = propensity_scores
        return treatments
    



def save_dataset(X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                Y0: np.ndarray = None, Y1: np.ndarray = None, 
                filename: str = "synthetic_continous_dataset.csv") -> pd.DataFrame:
 
    n_features = X.shape[1]
    feature_names = [f"x{i}" for i in range(n_features)]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['treatment'] = A
    df['outcome'] = Y
    
    if Y0 is not None:
        df['y0'] = Y0
    if Y1 is not None:
        df['y1'] = Y1
    
    if Y0 is not None and Y1 is not None:
        df['ite'] = Y1 - Y0
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    
    return df

def generate_single_dataset(dataset_id: int, num_samples: int = 1024, num_features: int = 10, 
                          output_dir: str = "data/continuous", seed_offset: int = 0):
    """
    Generate a single synthetic causal dataset.
    
    Args:
        dataset_id: ID for this dataset
        num_samples: Number of samples to generate
        num_features: Number of features to generate
        output_dir: Directory to save the dataset
        seed_offset: Offset for random seed to ensure different datasets
    """
    # Set unique random seed for each dataset
    np.random.seed(dataset_id + seed_offset)
    
    print(f"\n=== Generating Dataset {dataset_id} ===")
    
    # Define prior distributions for covariates
    def prior_layers_x():
        return np.random.randint(3, 7) 
    
    def prior_hidden_size_x():
        return np.random.randint(15, 40)  
    
    def prior_weight_x():
        return np.random.normal(0, 1)  
    
    # Create the DAG-structured SCM for covariates
    dag_scm = DAGStructuredSCM(
        prior_layers=prior_layers_x,
        prior_hidden_size=prior_hidden_size_x,
        prior_weight=prior_weight_x,
        edge_drop_prob=0.5,
        activation=lambda x: np.tanh(x)
    )
    
    # Generate the covariate dataset X
    X = dag_scm.generate_dataset(num_features, num_samples)
    

    def prior_layers_a():
        return np.random.randint(3, 5)  
    
    def prior_hidden_size_a():
        return np.random.randint(8, 20)  
    
    def prior_weight_a():
        return np.random.normal(0, 0.8)  
    

    treatment_assigner = TreatmentAssigner(
        prior_layers=prior_layers_a,
        prior_hidden_size=prior_hidden_size_a,
        prior_weight=prior_weight_a,
        activation=lambda x: np.tanh(x)
    )
    
    A = treatment_assigner.generate_treatments(X)
    

    def prior_layers_y():
        return np.random.randint(3, 6)  
    
    def prior_hidden_size_y():
        return np.random.randint(10, 25)  
    
    def prior_weight_y():
        return np.random.normal(0, 1.0)  
    
    outcome_generator = OutcomeGenerator(
        prior_layers=prior_layers_y,
        prior_hidden_size=prior_hidden_size_y,
        prior_weight=prior_weight_y,
        edge_drop_prob=0.4,
        activation=lambda x: np.tanh(x)
    )
    

    Y, Y0, Y1 = outcome_generator.generate_outcomes(X, A)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataset
    filename = os.path.join(output_dir, f"synthetic_continous_dataset_{dataset_id}.csv")
    dataset = save_dataset(X, A, Y, Y0, Y1, filename)
    
    return dataset, X, A, Y, Y0, Y1

def generate_multiple_datasets(num_datasets: int = 10, num_samples: int = 1024, 
                             num_features: int = 10, output_dir: str = "data/continuous",
                             base_seed: int = 42):
    """
    Generate multiple synthetic causal datasets.
    
    Args:
        num_datasets: Number of datasets to generate
        num_samples: Number of samples per dataset
        num_features: Number of features per dataset
        output_dir: Directory to save datasets
        base_seed: Base seed for reproducibility
    """
    print(f"Generating {num_datasets} datasets with {num_samples} samples each...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, num_datasets + 1):
        try:
            dataset, X, A, Y, Y0, Y1 = generate_single_dataset(
                dataset_id=i,
                num_samples=num_samples,
                num_features=num_features,
                output_dir=output_dir,
                seed_offset=base_seed
            )
            
        except Exception as e:
            print(f"Error generating dataset {i}: {str(e)}")
            continue
    

# Main execution
if __name__ == "__main__":
    # Generate 10k datasets with 1024 samples each
    generate_multiple_datasets(
        num_datasets=10,
        num_samples=1024,
        num_features=10,
        output_dir="DATA_standard/training_data",
        base_seed=42
    )
    
