
import torch 
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import fileinput
import math
import re

from transformers import AutoModel, AutoTokenizer, BertModel
from torch.nn.functional import cosine_similarity

from . import utils


class EvoFeatureGenerator:
    """A helper class to manage the Evo model and feature generation."""
    def __init__(self, model_name, device):
        print(f"Initializing EvoFeatureGenerator with model: {model_name} on device: {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Using AutoModel to get hidden states without the language modeling head
        self.model = BertModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    @torch.no_grad()
    def get_embedding(self, dna_sequence):
        """Generates an embedding for a single DNA sequence, handling long sequences by chunking."""
        if not dna_sequence:
            return np.zeros(self.embedding_dim)

        # --- FIX: Process long sequences in chunks ---
        
        # Tokenize the entire sequence once
        token_ids = self.tokenizer.encode(dna_sequence, add_special_tokens=False)
        
        # If the sequence is short enough, process it in one go
        max_length = 510 # Use 510 to leave space for [CLS] and [SEP] tokens
        if len(token_ids) <= max_length:
            inputs = self.tokenizer(dna_sequence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        # For long sequences, process in chunks and average the results
        chunk_embeddings = []
        for i in range(0, len(token_ids), max_length):
            chunk = token_ids[i:i + max_length]
            
            # Convert token ids back to string for the tokenizer
            chunk_sequence = self.tokenizer.decode(chunk)
            
            inputs = self.tokenizer(chunk_sequence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            
            # Get the mean embedding for this chunk and store it
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
            chunk_embeddings.append(chunk_embedding)

        # Average the embeddings of all chunks to get the final representation
        final_embedding = torch.stack(chunk_embeddings).mean(dim=0)
        return final_embedding.numpy()
    
    @torch.no_grad()
    def get_embeddings_batch(self, sequences: list[str]):
        """
        Generates embeddings for a BATCH of DNA sequences.
        Short sequences are batched in mini-batches. Long sequences are processed individually.
        """
        max_length = 510
        results = [None] * len(sequences)
        
        short_sequences_indices = []
        short_sequences_list = []
        long_sequences_indices = []

        # First, classify sequences as short or long based on token length
        for i, seq in enumerate(sequences):
            if not seq:
                results[i] = torch.zeros(self.embedding_dim, dtype=torch.float32)
                continue
            
            token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
            if len(token_ids) <= max_length:
                short_sequences_indices.append(i)
                short_sequences_list.append(seq)
            else:
                long_sequences_indices.append(i)

        # --- FIX: Process the batch of short sequences in smaller MINI-BATCHES ---
        if short_sequences_list:
            print(f"  > Processing {len(short_sequences_list)} short sequences in mini-batches...")
            mini_batch_size = 128  # You can adjust this size based on your system's memory
            
            for i in range(0, len(short_sequences_list), mini_batch_size):
                # Get the current mini-batch of sequences and their original indices
                batch_sequences = short_sequences_list[i:i + mini_batch_size]
                batch_indices = short_sequences_indices[i:i + mini_batch_size]
                
                print(f"    - Processing mini-batch {i//mini_batch_size + 1}/{(len(short_sequences_list) + mini_batch_size - 1)//mini_batch_size}...", end='\r', flush=True)

                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs)
                short_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
                
                # Place the results from the mini-batch into the correct final positions
                for j, original_index in enumerate(batch_indices):
                    results[original_index] = short_embeddings[j]
            print("\n  > Finished processing short sequences.")
                
        # Process long sequences one by one using the existing chunking method
        if long_sequences_indices:
            total_long = len(long_sequences_indices)
            print(f"  > Processing {total_long} long sequences individually...")
            for i, original_index in enumerate(long_sequences_indices):
                print(f"    - Processing long sequence {i+1}/{total_long}...", end='\r', flush=True)
                long_sequence = sequences[original_index]
                embedding_np = self.get_embedding(long_sequence)
                results[original_index] = torch.from_numpy(embedding_np)
            print("\n  > Finished processing long sequences.")

        # Ensure all results are filled and stack them into a single tensor
        for i, res in enumerate(results):
            if res is None:
                 results[i] = torch.zeros(self.embedding_dim, dtype=torch.float32)
                 
        final_embeddings = torch.stack(results)
        return final_embeddings
    

class Dataset_Pytorch(InMemoryDataset):
    def __init__(self, root, file_prefix, train_file_list, parameters):
        self.file_prefix = file_prefix
        self.train_file_list = train_file_list
        self.parameters = parameters

        super().__init__(root)
        
        loaded_data = torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices, self.G, self.node_list = loaded_data

    @property
    def processed_file_names(self):
        return ["all_graphs.pt"]

    def process(self):
        self.G = read_graph_set(self.file_prefix, self.train_file_list, self.parameters['minimum_contig_length'])

        self.node_list = list(self.G)


        # --- NEW: CONDITIONAL FEATURE GENERATION ---
        if self.parameters['feature_generation_method'] == 'evo':
            # --- EVO PATH ---
            print("Using 'evo' feature generation method.")
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            evo_gen = EvoFeatureGenerator(self.parameters['evo_model_name'], device)

            total_nodes = len(self.node_list)
            
            # 1. Collect all sequences for batch processing
            all_sequences = [self.G.nodes[node_id].get("sequence", "") for node_id in self.node_list]
            
            # 2. Generate embeddings in a batch
            print("Generating Evo embeddings for all contigs...")
            all_embeddings_tensor = evo_gen.get_embeddings_batch(all_sequences)

            # 3. Assign embeddings back to the nodes in the graph
            for i, node_id in enumerate(self.node_list):
                # We store the embedding as a tensor on the node
                self.G.nodes[node_id]["evo_embedding"] = all_embeddings_tensor[i]

            print("\nEvo embedding generation complete.")

            # 2. Calculate edge attributes using cosine similarity of embeddings
            print("Calculating edge attributes using cosine similarity...")
            for u, v in self.G.edges():
                emb_u = self.G.nodes[u]["evo_embedding"]
                emb_v = self.G.nodes[v]["evo_embedding"]
                # Add a dimension to make it (1, dim) for cosine_similarity
                similarity = cosine_similarity(emb_u.unsqueeze(0), emb_v.unsqueeze(0)).item()
                self.G.edges[u, v]["embedding_cosine_similarity"] = similarity
            
            # 3. Define the hybrid feature set for nodes
            # These are the original features you want to keep
            original_features_to_keep = ['coverage_norm', 'gc_norm', 'degree', 'length_norm']
            self.parameters['features'] = tuple(original_features_to_keep) + tuple([f'evo_{i}' for i in range(evo_gen.embedding_dim)])
            
            # 4. Construct the node feature matrix `x`
            x_list = []
            for node_id in self.node_list:
                original_feats = [self.G.nodes[node_id][f] for f in original_features_to_keep]
                evo_embedding = self.G.nodes[node_id]["evo_embedding"].numpy()
                hybrid_vector = np.concatenate([original_feats, evo_embedding])
                x_list.append(hybrid_vector)
            x = np.array(x_list)
            
            # 5. Construct edge attributes
            edge_attr_list = []
            for u, v, edge_data in self.G.edges(data=True):
                similarity = edge_data.get("embedding_cosine_similarity", 0.0)
                edge_attr_list.append([similarity])
                edge_attr_list.append([similarity]) # For the reverse edge

        else:
            # --- K-MER PATH (Existing logic) ---
            print("Using 'kmer' feature generation method.")
            for u, v in self.G.edges():
                kmer_u = np.array(self.G.nodes[u]["kmer_counts_norm"])
                kmer_v = np.array(self.G.nodes[v]["kmer_counts_norm"])
                dot_product = np.dot(kmer_u, kmer_v)
                self.G.edges[u, v]["kmer_dot_product"] = dot_product
            
            features = self.parameters["features"]
            x = np.array([[self.G.nodes[node_id][f] for f in features] for node_id in self.node_list])

            edge_attr_list = []
            for u, v, edge_data in self.G.edges(data=True):
                dot_product = edge_data.get("kmer_dot_product", 0.0)
                edge_attr_list.append([dot_product])
                edge_attr_list.append([dot_product])

        
        # --- NEW CODE TO CREATE THE BATCH TENSOR ---
        # 1. Get unique sample IDs, which correspond to the original graphs
        sample_ids = [node.split(':')[0] for node in self.node_list]
        unique_samples = sorted(list(set(sample_ids)))
        sample_to_idx = {sample_id: i for i, sample_id in enumerate(unique_samples)}
        
        # 2. Create the batch list by mapping each node to its graph's index
        batch_indices = [sample_to_idx[node.split(':')[0]] for node in self.node_list]
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long)


        # Get labels (y)
        label_features = ["plasmid_label", "chrom_label"]
        y = np.array([[self.G.nodes[node_id][f] for f in label_features] for node_id in self.node_list])
        if self.parameters["loss_function"] == "squaredhinge":
            y = y * 2 - 1

        # Get edge index
        node_to_idx = {node_id: i for i, node_id in enumerate(self.node_list)}
        edge_list_sources = []
        edge_list_targets = []
        for u, v in self.G.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_list_sources.extend([u_idx, v_idx])
            edge_list_targets.extend([v_idx, u_idx])

        edge_index = torch.tensor(np.vstack((edge_list_sources, edge_list_targets)), dtype=torch.long)
        
        # Convert the correctly generated edge_attr_list to a tensor
        edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float)


        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.float),
            edge_attr=edge_attr_tensor,
            batch=batch_tensor
        )
        
        processed_data, slices = self.collate([data])
        torch.save((processed_data, slices, self.G, self.node_list), self.processed_paths[0])


def KL(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def weighted_median(values, weights):
    middle = np.sum(weights) / 2
    cum = np.cumsum(weights)
    for (i, x) in enumerate(cum):
        if x >= middle:
            return values[i]
    assert False

def add_normalized_coverage(graph, current_nodes):
    """Add attribute coverage_norm which is original coverage divided by median weighted by length.
    (only for nodes in current_nodes list)"""

    # similarly to Unicycler's computation
    # from function get_median_read_depth in 
    # https://github.com/rrwick/Unicycler/blob/main/unicycler/assembly_graph.py

    sorted_nodes = sorted(current_nodes, key=lambda x : graph.nodes[x]["coverage"])
    lengths = np.array([graph.nodes[x]["length"] for x in sorted_nodes])
    coverages = np.array([graph.nodes[x]["coverage"] for x in sorted_nodes])
    median = weighted_median(coverages, lengths)
    for node_id in current_nodes:
        graph.nodes[node_id]["coverage_norm"] = graph.nodes[node_id]["coverage"] / median

def get_node_coverage(gfa_arguments, seq_length):
    """Return coverage parsed from dp or estimated from KC tag. 
    The second return value is True for dp and False for KC"""
    # try finding dp tag
    for x in gfa_arguments:
        match =  re.match(r'^dp:f:(.*)$',x)
        if match :
            return (float(match.group(1)), True)
    # try finding KC tag
    for x in gfa_arguments:
        match =  re.match(r'^KC:i:(.*)$',x)
        if match :
            return (float(match.group(1)) / seq_length, False)
    raise AssertionError("depth not found")

def read_graph(graph_file, csv_file, sample_id, graph, minimum_contig_length):
    """Read a single graph from gfa of gfa.gz, compute attributes, add its nodes and edges to nx graph. 
    Label csv file can be set to None. Contigs shorter than minimum_contig_length are contracted."""

    # first pass: read all nodes
    current_nodes = []
    whole_seq = ""  # concatenated contigs
    coverage_types = {True:0, False:0}  # which coverage types for individual nodes

    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file: 
            if isinstance(line, bytes):
                line = line.decode("utf-8") # convert byte sequences to strings
            parts = line.strip().split("\t")
            if parts[0] == "S":  # node line
                node_id = utils.get_node_id(sample_id, parts[1])
                seq = parts[2].upper()
                assert node_id not in graph, f"Duplicate node ID found: {node_id}"
                graph.add_node(node_id)
                
                # --- Now you can safely modify its attributes ---
                graph.nodes[node_id]["sequence"] = seq
                
                if not re.match(r'^[A-Z]*$', seq):
                    raise AssertionError(f"Bad sequence in {node_id}")

                whole_seq += "N" + seq
                current_nodes.append(node_id)
                seq_length = len(seq)

                graph.nodes[node_id]["contig"] = parts[1]
                graph.nodes[node_id]["sample"] = sample_id
                graph.nodes[node_id]["length"] = seq_length
                (coverage, is_dp) = get_node_coverage(parts[3:], seq_length)
                graph.nodes[node_id]["coverage"] = coverage
                coverage_types[is_dp] += 1
                graph.nodes[node_id]["gc"] = utils.get_gc_content(seq)
                graph.nodes[node_id]["kmer_counts_norm"] = utils.get_kmer_distribution(seq, scale=True)
    
    # check that only one coverage type seen
    assert coverage_types[True] == 0 or coverage_types[False] == 0

    # second pass: read all edges
    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file: 
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            parts = line.strip().split("\t")
            if parts[0] == "L":  # edge line
                graph.add_edge(utils.get_node_id(sample_id, parts[1]),
                               utils.get_node_id(sample_id, parts[3]))


    # get graph degrees
    for node_id in current_nodes:
            graph.nodes[node_id]["degree"] = graph.degree[node_id]

    # get gc of whole seq
    gc_of_whole_seq = utils.get_gc_content(whole_seq)
    # set normalized gc content
    for node_id in current_nodes:
        graph.nodes[node_id]["gc_norm"] =  graph.nodes[node_id]["gc"] - gc_of_whole_seq
    

    # get max length
    max_contig_length = max([graph.nodes[node_id]["length"] for node_id in current_nodes])
    # get normalized contig lengths (divided by max length)
    for node_id in current_nodes:
        #graph.nodes[node_id]["length_norm"] =  graph.nodes[node_id]["length"] / max_contig_length
        graph.nodes[node_id]["length_norm"] =  graph.nodes[node_id]["length"] / 2000000
        graph.nodes[node_id]["loglength"] = math.log(graph.nodes[node_id]["length"]+1)

    add_normalized_coverage(graph, current_nodes)
        

    # Each element is a float representing the proportion of a specific canonical k-mer
    all_kmer_counts_norm = np.array(utils.get_kmer_distribution(whole_seq, scale=True))

    for node_id in current_nodes:
        # Each element represents the difference in proportion for a specific k-mer between the individual contig and the whole sample
        diff = np.array(graph.nodes[node_id]["kmer_counts_norm"]) - all_kmer_counts_norm

        # quantifying the overall magnitude of the difference between the contig's k-mer profile and the whole sample's k-mer profile
        graph.nodes[node_id]["kmer_dist"] = np.linalg.norm(diff)

        # measures the similarity between the two k-mer distributions
        graph.nodes[node_id]["kmer_dot"] = np.dot(np.array(graph.nodes[node_id]["kmer_counts_norm"]),all_kmer_counts_norm)

        # KL divergence between the contig's k-mer distribution (a) and the whole sample's k-mer distribution (b)
        graph.nodes[node_id]["kmer_kl"] = KL(np.array(graph.nodes[node_id]["kmer_counts_norm"]),all_kmer_counts_norm)
    
        
    # read and add node labels
    if csv_file is not None:
        df_labels = pd.read_csv(csv_file)
        df_labels["id"] = df_labels["contig"].map(lambda x : utils.get_node_id(sample_id, x))
        df_labels.set_index("id", inplace=True)
    else:
        df_labels = pd.DataFrame()

    for node_id in current_nodes:
        label = None
        if node_id in df_labels.index:
            label = df_labels.loc[node_id, "label"]  # textual label

        pair = utils.label_to_pair(label)  # pair of binary values
        graph.nodes[node_id]["text_label"] = utils.pair_to_label(pair)
        graph.nodes[node_id]["plasmid_label"] = pair[0]
        graph.nodes[node_id]["chrom_label"] = pair[1]

    if minimum_contig_length > 0:
        delete_short_contigs(graph, current_nodes, minimum_contig_length)
    
def delete_short_contigs(graph, node_list, minimum_contig_length):
    """check length attribute of all contigs in node_list 
    and if some are shorter than minimum_contig_length,
    remove them from the graph and connect new neighbors"""
    for node_id in node_list:
        if graph.nodes[node_id]["length"] < minimum_contig_length:
            neighbors = list(graph.neighbors(node_id))
            all_new_edges = list(itertools.combinations(neighbors, 2))
            for edge in all_new_edges:
                graph.add_edge(edge[0], edge[1])
            graph.remove_node(node_id)


def read_single_graph(file_prefix, gfa_file, sample_id, minimum_contig_length):
    """Read single graph without node labels for testing"""
    graph = nx.Graph()
    graph_file = file_prefix + gfa_file
    read_graph(graph_file, None, sample_id, graph, minimum_contig_length)
    return graph

def read_graph_set(file_prefix, file_list, minimum_contig_length, read_labels=True):
    """Read several graph files to a single graph. 
    Node labels will be read from the csv file for each graph if read_labels is True. 
    Nodes shorter than minimum_contig_length will be deleted from the graph.
    """

    # read data frame with files
    train_files = pd.read_csv(file_list, names=('graph','csv','sample_id'))

    total_graphs = len(train_files)

    graph = nx.Graph()
    for idx, row in train_files.iterrows():

        print(f"Processing graph {idx + 1}/{total_graphs}: {row['sample_id']}")

        graph_file = file_prefix + row['graph']
        if read_labels:
            csv_file = file_prefix + row['csv']
        else:
            csv_file = None
        read_graph(graph_file, csv_file, row['sample_id'], graph, minimum_contig_length)

    return graph
