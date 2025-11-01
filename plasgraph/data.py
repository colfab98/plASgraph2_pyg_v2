import fileinput
import itertools
import math
import re
import subprocess
import os
import time

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


# prevent deadlocks with hugging face tokenizers when using multiple dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EdgeDataset(Dataset):
    """simple Dataset wrapper for a list of edges"""
    def __init__(self, edges):
        self.edges = edges  # store the list of edges

    def __len__(self):
        return len(self.edges)  # return the total number of edges

    def __getitem__(self, idx):
        return self.edges[idx]  # retrieve a single edge by its index


class SequenceDataset(Dataset):
    """
    dataset wrapper for sequences and their original indices
    this is crucial for mapping results (e.g., embeddings) back to the correct nodes
    """
    def __init__(self, sequences, original_indices):
        self.sequences = sequences      # store the list of dna sequences
        self.original_indices = original_indices    # store their corresponding original positions

    def __len__(self):
        return len(self.sequences)  # return the total number of sequences

    def __getitem__(self, idx):
        # return the sequence and its original index as a tuple
        return self.sequences[idx], self.original_indices[idx]

def sequence_collate_fn(batch):
    """
    custom collate function to organize a batch of (sequence, index) pairs
    it transforms a list of tuples into two separate lists
    """
    sequences = [item[0] for item in batch]     # unzip all sequences into a single list
    indices = [item[1] for item in batch]       # unzip all indices into a single list
    return sequences, indices



class EmbFeatureGenerator:
    """
    manages the dna foundation model for generating sequence embeddings.
    handles model loading, device placement, and efficient batch processing of sequences
    of any length, with support for multi-gpu inference via accelerate
    """
    def __init__(self, model_name, accelerator: Accelerator):
        print(f"Initializing DNABertFeatureGenerator with model: {model_name} on device: {accelerator.device}")
        self.accelerator = accelerator
        # load the pre-trained tokenizer; `trust_remote_code` is needed for custom model architectures
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # load the pre-trained model weights; `use_safetensors` is a secure and efficient format
        self.model = BertModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
        # get the size of the embeddings
        self.embedding_dim = self.model.config.hidden_size
        # prepare the model for the appropriate device (CPU/GPU) using the Accelerator
        self.model = self.accelerator.prepare(self.model)
        # set the model to evaluation mode to disable dropout and other training-specific layers
        self.model.eval()

    @torch.no_grad()    # disables gradient calculations to save memory and speed up inference

    def get_embedding(self, dna_sequence):
        """
        generates an embedding for a SINGLE dna sequence
        handles long sequences by chunking and averaging the results
        """
        # return a zero vector if the sequence is empty
        if not dna_sequence:
            return torch.zeros(self.embedding_dim, device=self.accelerator.device)

        # convert the dna string into a list of integer token IDs
        token_ids = self.tokenizer.encode(dna_sequence, add_special_tokens=False)      
        # set the maximum number of tokens the model can handle at once, leaving space for [CLS] and [SEP]
        max_length = 510 
        
        # if the sequence fits, process it in one go
        if len(token_ids) <= max_length:
            inputs = self.tokenizer(dna_sequence, return_tensors="pt")
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
            # pass the inputs to the model
            outputs = self.model(**inputs)
            # final embedding is the mean of all token hidden states from the last layer
            return outputs.last_hidden_state.mean(dim=1).squeeze()

        # for long sequences, process in chunks and average the results
        chunk_embeddings = []
        # iterate over the token IDs in steps of `max_length`
        for i in range(0, len(token_ids), max_length):
            chunk = token_ids[i:i + max_length]
            # manually create the input tensor for the chunk, adding the CLS and SEP token IDs
            input_ids = torch.tensor([self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]).unsqueeze(0)
            # create an attention mask of all ones
            attention_mask = torch.ones_like(input_ids)

            # pass tensors to the model on the correct device
            outputs = self.model(
                input_ids=input_ids.to(self.accelerator.device),
                attention_mask=attention_mask.to(self.accelerator.device)
            )
            
            # calculate the embedding for this chunk
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            chunk_embeddings.append(chunk_embedding)

        # stack all chunk embeddings into a single tensor and average them
        final_embedding = torch.stack(chunk_embeddings).mean(dim=0)
        return final_embedding
    
    @torch.no_grad()
    def get_embeddings_batch(self, sequences: list[str]):
        """
        generates embeddings for a BATCH of dna sequences with high efficiency using a hybrid strategy
        """
        max_length = 510
        # to hold final embeddings, ensuring order is preserved
        results = [None] * len(sequences)
        # lists to categorize sequences for different processing strategies
        short_sequences_indices = []
        short_sequences_list = []
        long_sequences_indices = []

        # iterate through all sequences to classify them as "short" or "long"
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

        # --- process all short sequences in efficient, padded batches ---
        if short_sequences_list:
            # define a mini-batch size for processing the short sequences
            mini_batch_size = 128
            short_seq_dataset = SequenceDataset(short_sequences_list, short_sequences_indices)
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

            for batch_sequences, batch_indices in progress_bar:
                # tokenize the entire batch at once, with padding and truncation
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
                
                # place each resulting embedding into the main `results` list at its correct original position
                for j, original_index in enumerate(batch_indices):
                    results[original_index] = short_embeddings[j]
                
        # --- process all long sequences by chunking and batching the chunks ---
        if long_sequences_indices:
            all_chunk_strings = []
            all_chunk_origins = []  

            # deconstruct all long sequences into a single flat list of string chunks
            for original_idx in long_sequences_indices:
                sequence = sequences[original_idx]
                # tokenize the long sequence
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                for i in range(0, len(token_ids), max_length):
                    chunk_token_ids = token_ids[i:i + max_length]
                    # decode the token chunk back to a string so the tokenizer can handle batch padding
                    chunk_string = self.tokenizer.decode(chunk_token_ids)
                    all_chunk_strings.append(chunk_string)
                    all_chunk_origins.append(original_idx)

            # dictionary to gather the embeddings for all chunks belonging to the same original sequence
            results_by_origin = {idx: [] for idx in long_sequences_indices}

            chunk_dataset = SequenceDataset(all_chunk_strings, all_chunk_origins)
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
    """
    main pytorch geometric dataset class that orchestrates the entire data pipeline
    it uses a caching mechanism: it processes raw data once, saves the result, and
    loads the cached version on subsequent runs for speed
    """
    def __init__(self, root, file_prefix, train_file_list, parameters, accelerator: Accelerator):
        self.file_prefix = file_prefix
        self.train_file_list = train_file_list
        self.parameters = parameters
        self.accelerator = accelerator

        # this call triggers the caching logic: if processed data is not found, `self.process()` is called
        super().__init__(root)
        
        # load the processed data from the cached file
        loaded_data = torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices, self.G, self.node_list = loaded_data

    @property
    def processed_file_names(self):
        """specifies the name of the cached file"""
        return ["all_graphs.pt"]

    def process(self):
        """
        the core data processing pipeline, called only when the cached file doesn't exist
        it handles file i/o, feature generation (potentially distributed), and final data object creation
        """
        all_embeddings_tensor = None
        is_distributed = self.accelerator.num_processes > 1     # check if we are in a multi-gpu setup

        # the main process (rank 0) is responsible for all file reading and initial data aggregation
        if self.accelerator.is_main_process:
            print("Main process: Reading raw graph files to generate sequence list...")

            # `read_graph_set` reads all gfa files and aggregates them into one large networkx graph
            self.G, train_files_df = read_graph_set(self.file_prefix, self.train_file_list, self.parameters, self.accelerator)
            self.node_list = list(self.G)

            # conditionally load edge read support counts if specified
            if self.parameters['dataset_type'] == 'new':
                self.accelerator.print("Loading edge read support features...")
                nx.set_edge_attributes(self.G, 0.0, 'read_support')
                # iterate through the manifest to find and read edge support files
                for _, row in tqdm(train_files_df.iterrows(), total=len(train_files_df), desc="  > Reading edge support files"):
                    edge_csv_path = self.file_prefix + row['edge_csv']
                    sample_id = row['sample_id']
                    if os.path.exists(edge_csv_path):
                        edge_df = pd.read_csv(edge_csv_path)
                        for _, edge_row in edge_df.iterrows():
                            u_contig = str(edge_row['contig_u'])
                            v_contig = str(edge_row['contig_v'])
                            node_u = utils.get_node_id(sample_id, u_contig)
                            node_v = utils.get_node_id(sample_id, v_contig)
                            
                            # update the 'read_support' attribute for the existing edge
                            if self.G.has_edge(node_u, node_v):
                                self.G.edges[node_u, node_v]['read_support'] = float(edge_row['total_support'])
            
            # extract all dna sequences from the graph nodes for feature generation
            all_sequences = [self.G.nodes[node_id].get("sequence", "") for node_id in self.node_list]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        start_time_emb = time.perf_counter()


        # --- generate sequence features (transformer embeddings) ---
        if self.parameters['feature_generation_method'] == 'emb':
            if is_distributed:
                # in a distributed setting, the main process broadcasts the sequence list to all others
                if self.accelerator.is_main_process:
                    objects_to_broadcast = [all_sequences]
                else:
                    objects_to_broadcast = [None]
                dist.broadcast_object_list(objects_to_broadcast, src=0)
                all_sequences = objects_to_broadcast[0]

            # each process computes embeddings for its assigned subset of sequences
            with self.accelerator.split_between_processes(all_sequences) as sequences_split:
                sequences_on_this_process = list(sequences_split)

                if sequences_on_this_process:
                    emb_gen = EmbFeatureGenerator(self.parameters['emb_model_name'], self.accelerator)
                    embeddings_on_this_process = emb_gen.get_embeddings_batch(sequences_on_this_process)
                else:
                    embeddings_on_this_process = torch.tensor([])

            if is_distributed:
                # gather the results from all processes
                gathered_embeddings = [None] * self.accelerator.num_processes
                # the main process concatenates the gathered tensors into one large tensor
                dist.all_gather_object(gathered_embeddings, embeddings_on_this_process)
                
                if self.accelerator.is_main_process:
                    # On the main process, concatenate the gathered tensors into one.
                    all_embeddings_tensor = torch.cat([t.to(self.accelerator.device) for t in gathered_embeddings if t.numel() > 0], dim=0)
            else:
                # if not distributed, the result is simply what the single process computed
                all_embeddings_tensor = embeddings_on_this_process
        
        elif self.parameters['feature_generation_method'] == 'kmer':
            pass # k-mer features are computed directly in `read_graph`
        else:
            raise ValueError(f"Unknown feature_generation_method: {self.parameters['feature_generation_method']}")

        end_time_emb = time.perf_counter()

        local_peak_mem_bytes = 0
        if torch.cuda.is_available():
            local_peak_mem_bytes = torch.cuda.max_memory_allocated(self.accelerator.device)
        
        # Create a tensor on the device for reduction
        local_peak_mem_tensor = torch.tensor(local_peak_mem_bytes, device=self.accelerator.device, dtype=torch.float)
        
        # Reduce across all processes to find the maximum peak
        global_peak_mem_tensor = self.accelerator.reduce(local_peak_mem_tensor, reduction='max')


        # --- final data assembly on the main process ---
        if self.accelerator.is_main_process:
            elapsed_seconds = end_time_emb - start_time_emb
            
            # --- THIS PART IS NEW ---
            peak_mem_gb = global_peak_mem_tensor.item() / (1024**3)
            self.accelerator.print("\n" + "="*60)
            self.accelerator.print(f"⏱️ [PERF] DNABERT embedding (data-parallel):")
            self.accelerator.print(f"  > Total Time: {elapsed_seconds:.2f} seconds")
            self.accelerator.print(f"  > Max Peak VRAM: {peak_mem_gb:.2f} GB")
            self.accelerator.print("="*60)
            # --- END OF NEW PART ---
            
            node_to_idx = {node_id: i for i, node_id in enumerate(self.node_list)}
            edge_list = list(self.G.edges())

            # --- calculate edge similarity ---
            if self.parameters['feature_generation_method'] == 'emb':
                u_indices = [node_to_idx[u] for u, v in edge_list]
                v_indices = [node_to_idx[v] for u, v in edge_list]
                u_indices_tensor = torch.tensor(u_indices, device=all_embeddings_tensor.device)
                v_indices_tensor = torch.tensor(v_indices, device=all_embeddings_tensor.device)
                emb_u = all_embeddings_tensor[u_indices_tensor]
                emb_v = all_embeddings_tensor[v_indices_tensor]
                # calculate cosine similarity between the two embeddings 
                similarities = F.cosine_similarity(emb_u, emb_v)
            
            elif self.parameters['feature_generation_method'] == 'kmer':
                self.accelerator.print("Calculating k-mer dot product for edges...")
                sim_list = []
                for u, v in tqdm(edge_list, desc="  > Calculating k-mer dot products"):
                    kmer_u = np.array(self.G.nodes[u]['kmer_counts_norm'])
                    kmer_v = np.array(self.G.nodes[v]['kmer_counts_norm'])
                    dot_product = np.dot(kmer_u, kmer_v)
                    sim_list.append(dot_product)
                similarities = torch.tensor(sim_list, dtype=torch.float)


            self.accelerator.print("Constructing final PyG Data object...")

            # --- construct edge attribute tensor ---
            edge_attr_list = []
            for i, (u, v) in enumerate(edge_list):
                similarity = similarities[i].item()
                if self.parameters['use_edge_read_counts']:
                    read_support = self.G.edges[u, v].get('read_support', 0.0)
                    # log-transform to handle zeros and compress the feature's range
                    log_read_support = math.log(read_support + 1.0)
                    feature_vector = [similarity, log_read_support]
                else:
                    # default behavior: use similarity only
                    feature_vector = [similarity]
                
                # append feature vector for both directions of the undirected edge
                edge_attr_list.extend([feature_vector, feature_vector])


            # construct node feature matrix (x)
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
       
            # use `inmemorydataset`'s collate method to prepare data for saving
            processed_data, slices = self.collate([data])
            # save the final processed data object and metadata to the cache file
            torch.save((processed_data, slices, self.G, self.node_list), self.processed_paths[0])
            self.accelerator.print("✅ Graph data processed and saved.")

        if is_distributed:
            # all processes wait here until the main process finishes saving the file
            self.accelerator.wait_for_everyone()



def KL(a, b):
    """calculates the kullback-leibler (kl) divergence between two distributions"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def weighted_median(values, weights):
    """computes the weighted median of a set of values"""
    middle = np.sum(weights) / 2
    cum = np.cumsum(weights)
    for (i, x) in enumerate(cum):
        if x >= middle:
            return values[i]
    assert False

def add_normalized_coverage(graph, current_nodes):
    """normalizes node coverage by the length-weighted median coverage of the sample"""
    sorted_nodes = sorted(current_nodes, key=lambda x : graph.nodes[x]["coverage"])
    lengths = np.array([graph.nodes[x]["length"] for x in sorted_nodes])
    coverages = np.array([graph.nodes[x]["coverage"] for x in sorted_nodes])
    median = weighted_median(coverages, lengths)
    for node_id in current_nodes:
        graph.nodes[node_id]["coverage_norm"] = graph.nodes[node_id]["coverage"] / median

def get_node_coverage(gfa_arguments, seq_length):
    """parses coverage from gfa tags, preferring 'dp' (depth) over 'kc' (k-mer count)"""
    for x in gfa_arguments:
        match =  re.match(r'^dp:f:(.*)$',x)
        if match :
            return (float(match.group(1)), True)
    for x in gfa_arguments:
        match =  re.match(r'^KC:i:(.*)$',x)
        if match :
            return (float(match.group(1)) / seq_length, False)
    raise AssertionError("depth not found")

def read_graph(graph_file, csv_file, sample_id, graph, minimum_contig_length):
    """
    parses a single gfa file, computes node attributes, adds nodes/edges to the main
    graph, and attaches labels from an optional csv file
    """
    current_nodes = []  
    whole_seq = ""   
    coverage_types = {True:0, False:0}  

    # first pass: read all nodes (segments 's') and compute initial features
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

    # second pass: read all edges (links 'l')
    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file: 
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            parts = line.strip().split("\t")

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
    """removes contigs shorter than a minimum length, connecting their neighbors"""
    for node_id in node_list:
        if graph.nodes[node_id]["length"] < minimum_contig_length:
            neighbors = list(graph.neighbors(node_id))
            all_new_edges = list(itertools.combinations(neighbors, 2))
            for edge in all_new_edges:
                graph.add_edge(edge[0], edge[1])
            graph.remove_node(node_id)


def read_single_graph(file_prefix, gfa_file, sample_id, minimum_contig_length):
    """convenience function to read a single, unlabeled graph for prediction"""
    graph = nx.Graph()
    graph_file = file_prefix + gfa_file
    read_graph(graph_file, None, sample_id, graph, minimum_contig_length)
    return graph

def read_graph_set(file_prefix, file_list, parameters, accelerator, read_labels=True):
    """reads a manifest file and aggregates all specified gfa files into one large networkx graph"""
    cols = ('gfa_gz','gfa_csv','edge_csv','sample_id') if parameters['dataset_type']=='new' else ('graph','csv','sample_id')
    all_files = pd.read_csv(file_list, names=cols)

    # filter files based on dataset type and assembly configuration
    if parameters['dataset_type'] == 'original':
        if parameters["assemblies"] == 'unicycler':
            train_files = all_files[all_files['graph'].str.contains('-u', na=False)].copy()
            if accelerator.is_main_process:
                print(f"Found {len(all_files)} total graphs, filtering to {len(train_files)} Unicycler graphs ('-u' in filename) for original dataset.")
        else: # 'all'
            train_files = all_files.copy()
            if accelerator.is_main_process:
                print(f"Found {len(all_files)} total graphs, using all of them as specified for original dataset.")
    else: # for the 'new' dataset type
        # we do not filter by filename and assume the manifest contains only the desired assemblies.
        train_files = all_files.copy()
        if accelerator.is_main_process:
            print(f"Found {len(train_files)} total graphs listed in the manifest for new dataset (no filename filtering).")
            
    total_graphs = len(train_files)
    graph = nx.Graph()

    for i, (idx, row) in enumerate(train_files.iterrows()):
        if accelerator.is_main_process:
            print(f"Processing graph {i + 1}/{total_graphs}: {row['sample_id']}")
        graph_file = file_prefix + (row['gfa_gz'] if parameters['dataset_type']=='new' else row['graph'])
        if read_labels:
            csv_file = file_prefix + (row['gfa_csv'] if parameters['dataset_type']=='new' else row['csv']) if read_labels else None
        else:
            csv_file = None
        read_graph(graph_file, csv_file, row['sample_id'], graph, parameters['minimum_contig_length'])

    return graph, train_files