import yaml

DEFAULT_FILENAME = "plasgraph_config.yaml"

PARAMS = [
    # Data and Feature Parameters

    {'name': 'features', 'type': str, 'default': 'coverage_norm,gc_norm,kmer_dot,degree,length_norm,loglength'}, # comma-separated string defining the node features to use from the graph data
    {'name': 'n_input_features', 'type': int, 'default': 6}, # total number of input node features, which must match the count in 'features'
    {'name': 'n_labels', 'type': int, 'default': 2}, # number of output classes for the classification task (e.g., 2 for plasmid/chromosome)
    {'name': 'minimum_contig_length', 'type': int, 'default': 100}, # shortest a contig can be to be included in the graph; shorter ones are removed
    {'name': 'feature_generation_method', 'type': str, 'default': 'evo'}, # method for generating sequence features; can be 'evo' for a Transformer model or 'kmer'
    {'name': 'evo_model_name', 'type': str, 'default': 'nvidia/Evo-2-8B-base'}, # specific name of the pre-trained Evo model to use if 'feature_generation_method' is 'evo'
    {'name': 'num_workers', 'type': int, 'default': 12}, # number of CPU worker processes to use for loading data in parallel

    # Model Architecture Parameters

    {'name': 'model_type', 'type': str, 'default': 'GGNNModel'}, # type of GNN architecture to use
    {'name': 'n_gnn_layers', 'type': int, 'default': 5}, # number of message-passing layers in the GNN
    {'name': 'n_channels_preproc', 'type': int, 'default': 10}, # feature dimension after the initial input preprocessing layer
    {'name': 'n_channels', 'type': int, 'default': 32}, # dimension of the hidden states within the main GNN layers
    {'name': 'preproc_activation', 'type': str, 'default': 'sigmoid'}, # activation function for the preprocessing layer
    {'name': 'gcn_activation', 'type': str, 'default': 'relu'}, # activation function used within the main GNN layers
    {'name': 'fully_connected_activation', 'type': str, 'default': 'relu'}, # activation function for the final dense layers before the output
    {'name': 'output_activation', 'type': str, 'default': 'none'}, # activation for the final output layer; 'none' is used to get raw logits for loss calculation
    {'name': 'tie_gnn_layers', 'type': bool, 'default': True}, # boolean that, if True, makes all GNN layers share the same weights
    {'name': 'neighbors_first_hop', 'type': int, 'default': 15}, # Number of neighbors to sample in the first GNN layer hop
    {'name': 'neighbors_subsequent_hops', 'type': int, 'default': 10}, # Number of neighbors to sample in all subsequent hops
    {'name': 'use_edge_gate', 'type': bool, 'default': True}, # whether to use an edge gating mechanism to modulate messages based on edge attributes
    {'name': 'edge_gate_hidden_dim', 'type': int, 'default': 32}, # for GGNNModel, this sets the hidden dimension of the neural network that computes edge weights
    {'name': 'edge_gate_depth', 'type': int, 'default': 3}, # for GGNNModel, this sets the number of layers in the edge gating network.
    {'name': 'use_GraphNorm', 'type': bool, 'default': True}, # whether to use GraphNorm for normalizing node features within each graph in a batch
    {'name': 'use_gru_update', 'type': bool, 'default': True}, # whether to use GRU-style update for the node features; if False, uses a simple linear transformation


    # Training and Optimization Parameters

    {'name': 'learning_rate', 'type': float, 'default': 0.05}, # initial learning rate for the optimizer
    {'name': 'epochs', 'type': int, 'default': 1500}, # maximum number of training epochs for the final model training run
    {'name': 'loss_function', 'type': str, 'default': 'crossentropy'}, # loss function to use for training
    {'name': 'plasmid_ambiguous_weight', 'type': float, 'default': 1.0}, # weight applied to "ambiguous" labels during loss calculation
    {'name': 'dropout_rate', 'type': float, 'default': 0.1}, # probability of dropout for regularization in the neural network layers
    {'name': 'l2_reg', 'type': float, 'default': 2.5e-4}, # strength of L2 weight decay regularization applied by the optimizer
    {'name': 'gradient_clipping', 'type': float, 'default': 0.0}, # maximum value to clip gradients to, which prevents exploding gradients; 0.0 disables it
    {'name': 'early_stopping_patience', 'type': int, 'default': 100}, # number of epochs to wait for validation loss to improve before stopping HPO trials
    {'name': 'early_stopping_patience_retrain', 'type': int, 'default': 100}, # patience for early stopping during the final model training run
    {'name': 'scheduler_patience', 'type': int, 'default': 10}, # number of epochs the learning rate scheduler will wait for improvement before reducing the LR
    {'name': 'scheduler_factor', 'type': float, 'default': 0.1}, # factor by which the learning rate is multiplied when the scheduler triggers
    {'name': 'random_seed', 'type': int, 'default': 123}, # random number generators to ensure run-to-run reproducibility
    {'name': 'batch_size', 'type': int, 'default': 64}, # number of graphs to process in a single batch during training
    
    # HPO and Evaluation Parameters

    {'name': 'k_folds', 'type': int, 'default': 5}, # number of folds to use for cross-validation during HPO and final training
    {'name': 'optuna_n_trials', 'type': int, 'default': 50}, # total number of HPO trials to run with Optuna
    {'name': 'epochs_trials', 'type': int, 'default': 100}, # maximum number of epochs to train for during a single HPO trial
    {'name': 'n_startup_trials', 'type': int, 'default': 20}, # number of initial HPO trials to run before the pruner can start stopping unpromising trials
    {'name': 'set_thresholds', 'type': bool, 'default': False}, # if True, enables the automatic determination of optimal classification thresholds
    {'name': 'plasmid_threshold', 'type': float, 'default': 0.5}, # classification probability threshold for a contig to be called a plasmid
    {'name': 'chromosome_threshold', 'type': float, 'default': 0.5}, # classification probability threshold for a contig to be called a chromosome

    ]


class config:
    """
    A configuration class to manage all hyperparameters and settings for the project.
    It loads default parameters and can override them with values from a YAML file.
    """

    def __init__(
        self,
        yaml_file=None
    ):
        # create list of parameter values and assign default value for each
        self._params = {}
        for param_dict in PARAMS:
            self._params[param_dict['name']] = param_dict['default']

        # read config file in YAML format and rewrite default values
        if yaml_file is not None:
            # read the whole file
            with open(yaml_file) as file:
                yaml_parameters = yaml.safe_load(file)
            # check each known parameter if it was in the file
            for param_dict in PARAMS:
                name = param_dict['name']
                if name in yaml_parameters:
                    # convert value to the correct type
                    value = yaml_parameters[name]
                    cast_value = param_dict['type'](value)
                    self._params[name] = cast_value
            known_params = {param_dict['name'] for param_dict in PARAMS}
            for name in yaml_parameters:
                if name not in known_params:
                    raise ValueError(f"Parameter {name} from {yaml_file} is not known")
        # special handling of features param (split to a list)
        self._params["features"] = tuple(self._params["features"].split(','))
        # special handling of output_activation - change to None if indicated
        if self._params["output_activation"].casefold() == "none".casefold():
            self._params["output_activation"] = None
    
    def __getitem__(self, key):
        return self._params[key]


    def __setitem__(self, key, value):
        assert key in self._params
        self._params[key] = value

    def write_yaml(self, filename):
        to_write = self._params.copy()

        # Only join 'features' if it's a list or tuple.
        if 'features' in to_write and isinstance(to_write['features'], (list, tuple)):
            to_write["features"] = ",".join(to_write["features"])

        with open(filename, "w") as file:
            yaml.dump(to_write, file, default_flow_style=False)

    
