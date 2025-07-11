
import torch 
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import fileinput
import math
import re
import subprocess

from transformers import AutoModel, AutoTokenizer, BertModel
from torch.nn.functional import cosine_similarity

from . import utils

from accelerate import Accelerator
import torch.distributed as dist
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset



class EdgeDataset(Dataset):
    """A simple Dataset to hold graph edges for batch processing."""
    def __init__(self, edges):
        self.edges = edges

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        return self.edges[idx]


class SequenceDataset(Dataset):
    """
    A simple Dataset to hold sequences and their original indices.
    
    This is a helper class used to wrap lists of sequences and their corresponding
    original positions, making them compatible with PyTorch's DataLoader for
    batch processing. This ensures that even after shuffling or batching, we
    can map processed results back to their original locations.
    """
    def __init__(self, sequences, original_indices):
        self.sequences = sequences
        self.original_indices = original_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.original_indices[idx]

def sequence_collate_fn(batch):
    """Custom collate function to batch sequences and their indices."""
    sequences = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    return sequences, indices



class EvoFeatureGenerator:
    """A helper class to manage the Evo model and feature generation."""
    def __init__(self, model_name, accelerator: Accelerator):
        print(f"Initializing EvoFeatureGenerator with model: {model_name} on device: {accelerator.device}")
        self.accelerator = accelerator
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Using AutoModel to get hidden states without the language modeling head
        self.model = BertModel.from_pretrained(model_name, trust_remote_code=True)
        self.embedding_dim = self.model.config.hidden_size

        self.model = self.accelerator.prepare(self.model)

        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, dna_sequence):
        """Generates an embedding for a single DNA sequence, handling long sequences by chunking."""
        if not dna_sequence:
            return torch.zeros(self.embedding_dim, device=self.accelerator.device)

        # --- FIX: Process long sequences in chunks ---
        
        # Tokenize the entire sequence once
        token_ids = self.tokenizer.encode(dna_sequence, add_special_tokens=False)
        
        # If the sequence is short enough, process it in one go
        max_length = 510 # Use 510 to leave space for [CLS] and [SEP] tokens
        if len(token_ids) <= max_length:
            inputs = self.tokenizer(dna_sequence, return_tensors="pt")
            # Move inputs to the correct device
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Return a GPU tensor, not a numpy array
            return outputs.last_hidden_state.mean(dim=1).squeeze()

        # For long sequences, process in chunks and average the results
        chunk_embeddings = []
        for i in range(0, len(token_ids), max_length):
            chunk = token_ids[i:i + max_length]

            # Use the correct variable 'chunk' and move tensors to the correct device
            input_ids = torch.tensor([self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            # Pass tensors to the model on the correct device
            outputs = self.model(
                input_ids=input_ids.to(self.accelerator.device),
                attention_mask=attention_mask.to(self.accelerator.device)
            )
            
            # Get the mean embedding for this chunk and store it
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            chunk_embeddings.append(chunk_embedding)

        # Average the embeddings of all chunks to get the final representation
        final_embedding = torch.stack(chunk_embeddings).mean(dim=0)
        return final_embedding
    
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
                results[i] = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.accelerator.device)
                continue
            
            token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
            if len(token_ids) <= max_length:
                short_sequences_indices.append(i)
                short_sequences_list.append(seq)
            else:
                long_sequences_indices.append(i)

        if short_sequences_list:
            mini_batch_size = 128
            
            short_seq_dataset = SequenceDataset(short_sequences_list, short_sequences_indices)
            data_loader = DataLoader(
                short_seq_dataset,
                batch_size=mini_batch_size,
                shuffle=False,  # No need to shuffle, just processing
                collate_fn=sequence_collate_fn
            )

            
            # Wrap the iterator with tqdm, which will only display on the main process
            progress_bar = tqdm(data_loader, 
                                desc="  > Processing Short Sequences", 
                                disable=not self.accelerator.is_main_process)
            
            for batch_sequences, batch_indices in progress_bar:
                inputs = self.tokenizer(
                    batch_sequences, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                short_embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Place results in their original positions
                for j, original_index in enumerate(batch_indices):
                    results[original_index] = short_embeddings[j]
                
        # ADD THIS BLOCK IN ITS PLACE
        if long_sequences_indices:
            all_chunk_strings = []
            all_chunk_origins = [] # Tracks which original sequence each chunk belongs to

            # 1. Deconstruct all long sequences into a single list of string chunks
            for original_idx in long_sequences_indices:
                sequence = sequences[original_idx]
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                for i in range(0, len(token_ids), max_length):
                    chunk_token_ids = token_ids[i:i + max_length]
                    # Decode back to a string so the tokenizer can handle batch padding
                    chunk_string = self.tokenizer.decode(chunk_token_ids)
                    all_chunk_strings.append(chunk_string)
                    all_chunk_origins.append(original_idx)

            # A dictionary to gather the chunk embeddings for each original sequence
            results_by_origin = {idx: [] for idx in long_sequences_indices}

            # 2. Use a DataLoader to create large batches of these chunks
            # We can reuse the SequenceDataset and collate_fn for this
            chunk_dataset = SequenceDataset(all_chunk_strings, all_chunk_origins)
            # Use a larger batch size for chunks as they are smaller
            chunk_loader = DataLoader(chunk_dataset, batch_size=256, collate_fn=sequence_collate_fn)

            progress_bar = tqdm(chunk_loader,
                                desc="  > Processing Long Sequence Chunks",
                                disable=not self.accelerator.is_main_process)

            # 3. Process the large batches of chunks on the GPU
            for chunk_strings_batch, origin_indices_batch in progress_bar:
                inputs = self.tokenizer(chunk_strings_batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                chunk_embeddings = outputs.last_hidden_state.mean(dim=1)

                # Gather the results
                for i, origin_idx in enumerate(origin_indices_batch):
                    results_by_origin[origin_idx].append(chunk_embeddings[i])

            # 4. Average the chunk embeddings for each original sequence
            for original_idx, gathered_chunks in results_by_origin.items():
                if gathered_chunks:
                    final_embedding = torch.stack(gathered_chunks).mean(dim=0)
                    results[original_idx] = final_embedding

        for i, res in enumerate(results):
            if res is None:
                 results[i] = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.accelerator.device)
                 
        final_embeddings = torch.stack(results)
        return final_embeddings
    

class Dataset_Pytorch(InMemoryDataset):
    def __init__(self, root, file_prefix, train_file_list, parameters, accelerator: Accelerator):
        self.file_prefix = file_prefix
        self.train_file_list = train_file_list
        self.parameters = parameters
        self.accelerator = accelerator

        # checks for a processed file in the 'root' directory and will call self.process() automatically if it's not found
        super().__init__(root)
        
        loaded_data = torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices, self.G, self.node_list = loaded_data

    # tells the InMemoryDataset framework what the final, cached file containing the processed data should be named
    @property
    def processed_file_names(self):
        return ["all_graphs.pt"]

    def process(self):

        # check ensures that the file I/O and initial data aggregation are
        # performed by only ONE process (the main process) in a distributed setup
        if self.accelerator.is_main_process:
            print("Main process: Reading raw graph files to generate sequence list...")

            # Read all specified graph files (GFAs) and combine them into a single NetworkX graph object
            self.G = read_graph_set(self.file_prefix, self.train_file_list, self.parameters['minimum_contig_length'], self.accelerator)
            self.node_list = list(self.G)
            
            # This is the data all other processes need
            all_sequences = [self.G.nodes[node_id].get("sequence", "") for node_id in self.node_list]
            
            # We put the list into a container to broadcast it
            objects_to_broadcast = [all_sequences]
            print(f"Main process: Broadcasting {len(all_sequences)} sequences to other processes...")
        else:
            # Other processes prepare an empty container to receive the data
            objects_to_broadcast = [None]

        # broadcast the data from the main process to all others 
        # the main process (src=0) sends the list of all sequences to every other process
        dist.broadcast_object_list(objects_to_broadcast, src=0)

        # all processes now have an identical copy of the sequence list
        all_sequences = objects_to_broadcast[0]

        all_embeddings_tensor = None
        if self.parameters['feature_generation_method'] == 'evo':
            self.accelerator.print("All processes: Starting parallel embedding generation.")

            # accelerator's context manager to automatically split the list of all sequences
            # into unique, evenly-sized chunks for each process
            with self.accelerator.split_between_processes(all_sequences) as sequences_split:

                # unique subset of sequences this specific process is responsible for
                sequences_on_this_process = list(sequences_split)

                evo_gen = EvoFeatureGenerator(self.parameters['evo_model_name'], self.accelerator)
                self.accelerator.print(f"Process {self.accelerator.process_index}: Generating {len(sequences_on_this_process)} embeddings.")
                embeddings_on_this_process = evo_gen.get_embeddings_batch(sequences_on_this_process)

                # gather the results from all processes
                gathered_embeddings = [None] * self.accelerator.num_processes
                dist.all_gather_object(gathered_embeddings, embeddings_on_this_process)

                # assemble the final tensor on the main process only
                if self.accelerator.is_main_process:
                    main_process_device = self.accelerator.device
                    tensors_on_main_device = [
                        t.to(main_process_device) for t in gathered_embeddings
                    ]

                    # concatenate the list of tensors into a single, complete tensor
                    all_embeddings_tensor = torch.cat(tensors_on_main_device, dim=0)


        # makes all processes wait at this line until every other process has also reached it
        self.accelerator.wait_for_everyone()

        # parallel edge attribute calculation
        if self.accelerator.is_main_process:
            # create a map from a node's string ID to its integer index in the tensor for efficient lookups
            node_to_idx = {node_id: i for i, node_id in enumerate(self.node_list)}
            edge_list = list(self.G.edges())

            emb_shape = all_embeddings_tensor.shape
            objects_to_broadcast = [node_to_idx, edge_list, emb_shape]
            self.accelerator.print(f"Broadcasting node map and {len(edge_list)} edges for parallel processing.")
        else:
            objects_to_broadcast = [None, None, None]

        # broadcast the list of Python objects from the main process (src=0) to all others.
        dist.broadcast_object_list(objects_to_broadcast, src=0)
        # unpack the received objects on all processes
        node_to_idx, edge_list, emb_shape = objects_to_broadcast

        # prepare the tensor for broadcasting on the sender and create empty receptacles on the receivers
        if self.accelerator.is_main_process:
            embeddings_to_broadcast = all_embeddings_tensor.contiguous()
        else:
            # create an empty tensor with the exact shape and type needed to receive the broadcasted data
            embeddings_to_broadcast = torch.empty(emb_shape, dtype=torch.float32, device=self.accelerator.device)

        # perform the broadcast
        dist.broadcast(embeddings_to_broadcast, src=0)
        # assign the now-populated tensor on each process to a new variable
        all_embeddings_tensor_local = embeddings_to_broadcast

        # automatically split the full edge_list into a unique, evenly-sized chunk for this specific process
        with self.accelerator.split_between_processes(edge_list) as edges_on_this_process:
            local_similarities = []
            if len(edges_on_this_process) > 0:

                # pre-calculate all indices on the CPU at once to minimize loop overhead
                u_indices_all = torch.tensor([node_to_idx[u] for u, v in edges_on_this_process], dtype=torch.long)
                v_indices_all = torch.tensor([node_to_idx[v] for u, v in edges_on_this_process], dtype=torch.long)
                
                # process in very large chunks to give the GPU a substantial amount of work
                chunk_size = 65536

                # progress bar
                loop_iterator = range(0, len(edges_on_this_process), chunk_size)
                progress_bar = tqdm(loop_iterator,
                                    desc="  > Calculating Edge Attributes",
                                    disable=not self.accelerator.is_main_process)
                
                # loop over the assigned edges in large chunks
                for i in range(0, len(edges_on_this_process), chunk_size):
                    # get the current chunk of indices and move it to the active GPU
                    u_indices_chunk = u_indices_all[i:i+chunk_size].to(self.accelerator.device)
                    v_indices_chunk = v_indices_all[i:i+chunk_size].to(self.accelerator.device)

                    # gather the corresponding embeddings directly on the GPU
                    emb_u_batch = all_embeddings_tensor_local[u_indices_chunk]
                    emb_v_batch = all_embeddings_tensor_local[v_indices_chunk]

                    # perform the fast GPU calculation on the large batch
                    similarities = cosine_similarity(emb_u_batch, emb_v_batch)
                    local_similarities.extend(similarities.cpu().tolist())

        # Gather results from all processes
        gathered_sims_nested = [None] * self.accelerator.num_processes
        dist.all_gather_object(gathered_sims_nested, local_similarities)
        
        # final graph construction (on main process)
        if self.accelerator.is_main_process:
            self.accelerator.print("Constructing final PyG Data object...")
            
            # flatten the gathered lists. order is preserved by split_between_processes
            flat_similarities = [item for sublist in gathered_sims_nested for item in sublist]

            # create the final edge attributes list for the PyG Data object
            edge_attr_list = []
            for sim in flat_similarities:
                # append for both directions of the undirected edge
                edge_attr_list.append([sim])
                edge_attr_list.append([sim])

            # construct node features (x) and batch tensor
            features = self.parameters["features"]
            x = np.array([[self.G.nodes[node_id][f] for f in features] for node_id in self.node_list])

            # construct the batch tensor, which maps each node to its original sample graph
            sample_ids = [node.split(':')[0] for node in self.node_list]
            unique_samples = sorted(list(set(sample_ids)))
            sample_to_idx_map = {sample_id: i for i, sample_id in enumerate(unique_samples)}
            batch_indices = [sample_to_idx_map[node.split(':')[0]] for node in self.node_list]
            batch_tensor = torch.tensor(batch_indices, dtype=torch.long)

            # construct the label matrix (y)
            label_features = ["plasmid_label", "chrom_label"]
            y = np.array([[self.G.nodes[node_id][f] for f in label_features] for node_id in self.node_list])

            # construct the edge index tensor in COO format [2, num_edges]
            edge_list_sources, edge_list_targets = [], []
            for u, v in edge_list:
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                # add entries for both directions of the edge
                edge_list_sources.extend([u_idx, v_idx])
                edge_list_targets.extend([v_idx, u_idx])
            edge_index = torch.tensor(np.vstack((edge_list_sources, edge_list_targets)), dtype=torch.long)
            
            # create the final PyG Data object
            data = Data(
                x=torch.tensor(x, dtype=torch.float),                       # node features
                edge_index=edge_index,                                      # graph connectivity
                y=torch.tensor(y, dtype=torch.float),                       # node lables
                edge_attr=torch.tensor(edge_attr_list, dtype=torch.float),  # edge features
                batch=batch_tensor                                          # maps nodes to samples
            )
            
            # use the parent class's collate method to prepare the data for batching
            processed_data, slices = self.collate([data])
            torch.save((processed_data, slices, self.G, self.node_list), self.processed_paths[0])
            self.accelerator.print("✅ Graph data processed and saved.")

        self.accelerator.wait_for_everyone()


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
    """
    Parses a single GFA file, computes node attributes, and adds its nodes and edges
    to a pre-existing NetworkX graph object.
    """

    # first pass: read all nodes and compute initial features
    current_nodes = [] # nodes from this specific file
    whole_seq = ""  # concatenated contigs
    coverage_types = {True:0, False:0}  # which coverage types for individual nodes

    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file: 
            if isinstance(line, bytes):
                line = line.decode("utf-8") # convert byte sequences to strings
            parts = line.strip().split("\t")

            # "S" lines represent Segments, which are the nodes (contigs) of the graph
            if parts[0] == "S": 
                # create a unique node ID by combining the sample ID and contig name
                node_id = utils.get_node_id(sample_id, parts[1])
                seq = parts[2].upper()
                assert node_id not in graph, f"Duplicate node ID found: {node_id}"

                # add the node to the main graph object passed into this function
                graph.add_node(node_id)
                graph.nodes[node_id]["sequence"] = seq
                if not re.match(r'^[A-Z]*$', seq):
                    raise AssertionError(f"Bad sequence in {node_id}")
                whole_seq += "N" + seq
                current_nodes.append(node_id)
                seq_length = len(seq)

                # populate the node's attribute dictionary with features
                graph.nodes[node_id]["contig"] = parts[1]
                graph.nodes[node_id]["sample"] = sample_id
                graph.nodes[node_id]["length"] = seq_length
                (coverage, is_dp) = get_node_coverage(parts[3:], seq_length)
                graph.nodes[node_id]["coverage"] = coverage
                coverage_types[is_dp] += 1
                graph.nodes[node_id]["gc"] = utils.get_gc_content(seq)
                graph.nodes[node_id]["kmer_counts_norm"] = utils.get_kmer_distribution(seq, scale=True)
    
    # code counts how many contigs in the file use the dp tag versus the KC tag. 
    # the assert statement then ensures that the file doesn't contain a mix of both. 
    # all contigs must use one method or the other
    assert coverage_types[True] == 0 or coverage_types[False] == 0

    # second pass: read all edges
    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file: 
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            parts = line.strip().split("\t")

            # "L" lines represent Links, which are the edges of the graph
            if parts[0] == "L":  
                graph.add_edge(utils.get_node_id(sample_id, parts[1]),
                               utils.get_node_id(sample_id, parts[3]))


    # calculate node degrees
    for node_id in current_nodes:
            graph.nodes[node_id]["degree"] = graph.degree[node_id]

    # calculate normalized GC content
    gc_of_whole_seq = utils.get_gc_content(whole_seq)
    for node_id in current_nodes:
        graph.nodes[node_id]["gc_norm"] =  graph.nodes[node_id]["gc"] - gc_of_whole_seq
    

    # calculate normalized and log-transformed length
    for node_id in current_nodes:
        # normalize length by a fixed large value to keep it in a reasonable range
        graph.nodes[node_id]["length_norm"] =  graph.nodes[node_id]["length"] / 2000000
        graph.nodes[node_id]["loglength"] = math.log(graph.nodes[node_id]["length"]+1)

    # calculate normalized coverage
    add_normalized_coverage(graph, current_nodes)
        

    # calculate advanced k-mer features based on the whole sample's k-mer profile
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
    
        
    # if a label file is provided, read it and attach labels to the nodes
    if csv_file is not None:
        df_labels = pd.read_csv(csv_file)
        df_labels["id"] = df_labels["contig"].map(lambda x : utils.get_node_id(sample_id, x))
        df_labels.set_index("id", inplace=True)
    else:
        df_labels = pd.DataFrame()

    for node_id in current_nodes:
        label = None
        if node_id in df_labels.index:
            label = df_labels.loc[node_id, "label"]  

        pair = utils.label_to_pair(label)  # pair of binary values
        graph.nodes[node_id]["text_label"] = utils.pair_to_label(pair)
        graph.nodes[node_id]["plasmid_label"] = pair[0]
        graph.nodes[node_id]["chrom_label"] = pair[1]

    # remove nodes that are shorter than the specified minimum length
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

def read_graph_set(file_prefix, file_list, minimum_contig_length, accelerator, read_labels=True):
    """
    Reads and aggregates several individual graph files into a single NetworkX graph.

    This function iterates through a manifest file that lists different samples,
    and for each sample, it calls a helper function (`read_graph`) to parse the
    corresponding graph file and add its contents to a unified graph object.
    """

    # read data frame with files
    train_files = pd.read_csv(file_list, names=('graph','csv','sample_id'))

    total_graphs = len(train_files)

    graph = nx.Graph()
    for idx, row in train_files.iterrows():

        if accelerator.is_main_process:
            print(f"Processing graph {idx + 1}/{total_graphs}: {row['sample_id']}")

        graph_file = file_prefix + row['graph']
        if read_labels:
            csv_file = file_prefix + row['csv']
        else:
            csv_file = None
        read_graph(graph_file, csv_file, row['sample_id'], graph, minimum_contig_length)

    return graph
