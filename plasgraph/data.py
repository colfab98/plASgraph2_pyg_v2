import fileinput
import itertools
import math
import re
import subprocess
import os

from accelerate import Accelerator
import networkx as nx
import numpy as np
import pandas as pd
import torch 
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel

from . import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        # load the tokenizer specific to the chosen Evo model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # load the base Evo model (BertModel) to get the raw hidden states
        self.model = BertModel.from_pretrained(model_name, trust_remote_code=True)
        # get the size of the embeddings
        self.embedding_dim = self.model.config.hidden_size
        # prepare the model for the appropriate device (CPU/GPU) using the Accelerator
        self.model = self.accelerator.prepare(self.model)
        # set the model to evaluation mode to disable dropout and other training-specific layers
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, dna_sequence):
        """Generates an embedding for a single DNA sequence, handling long sequences by chunking."""
        # if the input sequence is empty, return a zero vector of the correct embedding dimension
        if not dna_sequence:
            return torch.zeros(self.embedding_dim, device=self.accelerator.device)
        # tokenize the entire DNA sequence into a list of token IDs
        token_ids = self.tokenizer.encode(dna_sequence, add_special_tokens=False)      
        # set the maximum number of tokens the model can handle at once, leaving space for [CLS] and [SEP]
        max_length = 510 
        if len(token_ids) <= max_length:
            # use the tokenizer to prepare the full input with special tokens and return PyTorch tensors
            inputs = self.tokenizer(dna_sequence, return_tensors="pt")
            # move inputs to the correct device
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
            # pass the inputs to the model
            outputs = self.model(**inputs)
            # calculate the final embedding by taking the mean of the last hidden state across all tokens and remove the batch dimension
            return outputs.last_hidden_state.mean(dim=1).squeeze()

        # for long sequences, process in chunks and average the results
        chunk_embeddings = []
        # iterate over the token IDs in steps of `max_length`
        for i in range(0, len(token_ids), max_length):
            # extract the current chunk of token IDs
            chunk = token_ids[i:i + max_length]
            # manually create the input tensor for the chunk, adding the CLS and SEP token IDs
            input_ids = torch.tensor([self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]).unsqueeze(0)
            # create an attention mask of all ones, as we want to attend to all tokens in the chunk
            attention_mask = torch.ones_like(input_ids)

            # pass tensors to the model on the correct device
            outputs = self.model(
                input_ids=input_ids.to(self.accelerator.device),
                attention_mask=attention_mask.to(self.accelerator.device)
            )
            
            # calculate the mean embedding for this chunk
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            chunk_embeddings.append(chunk_embedding)

        # average the embeddings of all chunks by stacking them into a tensor and taking the mean along the new dimension
        final_embedding = torch.stack(chunk_embeddings).mean(dim=0)
        return final_embedding
    
    @torch.no_grad()
    def get_embeddings_batch(self, sequences: list[str]):
        """
        Generates embeddings for a BATCH of DNA sequences.
        Short sequences are batched together, while long sequences are chunked and processed efficiently.
        """
        max_length = 510
        # to store the final embeddings, ensuring the order is preserved
        results = [None] * len(sequences)
        # lists to hold the indices and content of short and long sequences
        short_sequences_indices = []
        short_sequences_list = []
        long_sequences_indices = []

        # first, classify all sequences as short or long based on their tokenized length
        for i, seq in enumerate(sequences):
            # handle empty sequences
            if not seq:
                results[i] = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.accelerator.device)
                continue
            # tokenize the sequence to find its length
            token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
            if len(token_ids) <= max_length:
                short_sequences_indices.append(i)
                short_sequences_list.append(seq)
            else:
                long_sequences_indices.append(i)

        # --- process all short sequences in batches ---
        if short_sequences_list:
            # define a mini-batch size for processing the short sequences
            mini_batch_size = 128
            # PyTorch Dataset to wrap the short sequences and their original indices
            short_seq_dataset = SequenceDataset(short_sequences_list, short_sequences_indices)
            # DataLoader to handle batching of the short sequences
            data_loader = DataLoader(
                short_seq_dataset,
                batch_size=mini_batch_size,
                shuffle=False,  
                # use a custom collate function to handle sequences and indices
                collate_fn=sequence_collate_fn 
            )

            progress_bar = tqdm(data_loader, 
                                desc="  > Processing Short Sequences", 
                                disable=not self.accelerator.is_main_process)
            # iterate over the mini-batches of short sequences
            for batch_sequences, batch_indices in progress_bar:
                # tokenize the entire batch at once, with padding and truncation
                inputs = self.tokenizer(
                    batch_sequences, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                # move the tokenized batch to the correct GPU/CPU
                inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
                # run the model on the batch
                outputs = self.model(**inputs)
                # calculate the mean embedding for every sequence in the batch
                short_embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # place each resulting embedding into the main `results` list at its correct original position
                for j, original_index in enumerate(batch_indices):
                    results[original_index] = short_embeddings[j]
                
        # --- process all long sequences by chunking and batching the chunks ---
        if long_sequences_indices:
            all_chunk_strings = []
            all_chunk_origins = [] # Tracks which original sequence each chunk belongs to

            # deconstruct all long sequences into a single flat list of string chunks
            for original_idx in long_sequences_indices:
                sequence = sequences[original_idx]
                # tokenize the long sequence
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                # iterate through the token IDs in chunks
                for i in range(0, len(token_ids), max_length):
                    chunk_token_ids = token_ids[i:i + max_length]
                    # decode the token chunk back to a string so the tokenizer can handle batch padding
                    chunk_string = self.tokenizer.decode(chunk_token_ids)
                    all_chunk_strings.append(chunk_string)
                    all_chunk_origins.append(original_idx)

            # dictionary to gather the embeddings for all chunks belonging to the same original sequence
            results_by_origin = {idx: [] for idx in long_sequences_indices}

            # DataLoader to create large batches of these chunks for efficient processing
            chunk_dataset = SequenceDataset(all_chunk_strings, all_chunk_origins)
            # use a larger batch size for chunks as they are uniformly sized and smaller
            chunk_loader = DataLoader(chunk_dataset, batch_size=256, collate_fn=sequence_collate_fn)

            progress_bar = tqdm(chunk_loader,
                                desc="  > Processing Long Sequence Chunks",
                                disable=not self.accelerator.is_main_process)

            # process the large batches of chunks on the GPU
            for chunk_strings_batch, origin_indices_batch in progress_bar:
                inputs = self.tokenizer(chunk_strings_batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                chunk_embeddings = outputs.last_hidden_state.mean(dim=1)

                # gather the results, appending each chunk's embedding to the list for its original sequence
                for i, origin_idx in enumerate(origin_indices_batch):
                    results_by_origin[origin_idx].append(chunk_embeddings[i])

            # average the chunk embeddings for each original sequence to get the final embedding
            for original_idx, gathered_chunks in results_by_origin.items():
                if gathered_chunks:
                    final_embedding = torch.stack(gathered_chunks).mean(dim=0)
                    results[original_idx] = final_embedding

        # final check for any sequences that might not have been processed
        for i, res in enumerate(results):
            if res is None:
                 results[i] = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.accelerator.device)
        # stack the list of individual tensors into a single batch tensor for the final output       
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

        is_distributed = self.accelerator.num_processes > 1

        # check ensures that the file I/O and initial data aggregation are
        # performed by only ONE process (the main process) in a distributed setup
        if self.accelerator.is_main_process:
            print("Main process: Reading raw graph files to generate sequence list...")

            # read all specified graph files (GFAs) and combine them into a single NetworkX graph object
            self.G = read_graph_set(self.file_prefix, self.train_file_list, self.parameters['minimum_contig_length'], self.accelerator)
            self.node_list = list(self.G)
            
            # data all other processes need
            all_sequences = [self.G.nodes[node_id].get("sequence", "") for node_id in self.node_list]

        if is_distributed:
            # Broadcast the initial data from the main process to all others
            if self.accelerator.is_main_process:
                objects_to_broadcast = [all_sequences]
            else:
                objects_to_broadcast = [None]
            dist.broadcast_object_list(objects_to_broadcast, src=0)
            all_sequences = objects_to_broadcast[0]

        with self.accelerator.split_between_processes(all_sequences) as sequences_split:
            sequences_on_this_process = list(sequences_split)
            
            # Check if there are sequences to process to avoid unnecessary initialization
            if sequences_on_this_process:
                evo_gen = EvoFeatureGenerator(self.parameters['evo_model_name'], self.accelerator)
                embeddings_on_this_process = evo_gen.get_embeddings_batch(sequences_on_this_process)
            else:
                embeddings_on_this_process = torch.tensor([])


        if is_distributed:
            # If distributed, gather the results from all processes.
            gathered_embeddings = [None] * self.accelerator.num_processes
            # This is the line that was failing. It's now safely inside the conditional block.
            dist.all_gather_object(gathered_embeddings, embeddings_on_this_process)
            
            if self.accelerator.is_main_process:
                # On the main process, concatenate the gathered tensors into one.
                all_embeddings_tensor = torch.cat([t.to(self.accelerator.device) for t in gathered_embeddings if t.numel() > 0], dim=0)
        else:
            # If not distributed, the results are simply what this one process computed.
            all_embeddings_tensor = embeddings_on_this_process


        if self.accelerator.is_main_process:
            node_to_idx = {node_id: i for i, node_id in enumerate(self.node_list)}
            edge_list = list(self.G.edges())

            # Calculate similarities on the main process
            u_indices = [node_to_idx[u] for u, v in edge_list]
            v_indices = [node_to_idx[v] for u, v in edge_list]
            
            # Move indices to the correct device for gathering embeddings
            u_indices_tensor = torch.tensor(u_indices, device=all_embeddings_tensor.device)
            v_indices_tensor = torch.tensor(v_indices, device=all_embeddings_tensor.device)

            emb_u = all_embeddings_tensor[u_indices_tensor]
            emb_v = all_embeddings_tensor[v_indices_tensor]

            # Calculate cosine similarity for all edges at once
            similarities = F.cosine_similarity(emb_u, emb_v)

            # ------------ THIS IS THE FIX ------------
            # REMOVE the dist.all_gather_object call and the list flattening.
            # We already have the final `similarities` tensor.
            # -----------------------------------------

            self.accelerator.print("Constructing final PyG Data object...")

            # Assign the calculated similarities back to the NetworkX graph object
            self.accelerator.print("Attaching edge attributes to NetworkX graph object...")
            for i, (u, v) in enumerate(edge_list):
                # Use the `similarities` tensor directly
                self.G.edges[u, v]['embedding_cosine_similarity'] = similarities[i].item()



        # create the final edge attributes list for the PyG Data object
        edge_attr_list = []
        # Correctly loop over the `similarities` tensor
        for sim in similarities: 
            # append for both directions of the undirected edge
            # Use .item() to get the Python number from the tensor
            edge_attr_list.append([sim.item()]) 
            edge_attr_list.append([sim.item()])

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

        if is_distributed:
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

    # Read all file entries from the manifest
    all_files = pd.read_csv(file_list, names=('graph','csv','sample_id'))

    # Filter the DataFrame to keep only rows where the 'graph' filename contains '-u'
    train_files = all_files[all_files['graph'].str.contains('-u', na=False)].copy()

    if accelerator.is_main_process:
        print(f"Found {len(all_files)} total graphs, filtered to {len(train_files)} graphs with '-u' in filename.")

    total_graphs = len(train_files)

    graph = nx.Graph()
    for i, (idx, row) in enumerate(train_files.iterrows()):

        if accelerator.is_main_process:
            print(f"Processing graph {i + 1}/{total_graphs}: {row['sample_id']}")

        graph_file = file_prefix + row['graph']
        if read_labels:
            csv_file = file_prefix + row['csv']
        else:
            csv_file = None
        read_graph(graph_file, csv_file, row['sample_id'], graph, minimum_contig_length)

    return graph
