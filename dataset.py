import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch.nn as nn
import anndata as ann
from collections import defaultdict
from torch.utils.data import Dataset,DataLoader,Sampler, WeightedRandomSampler

### below class is used to load the single cell data from the h5ad files
### specifically for the macrogene aae model        
class SingleCellDataset(Dataset):
    """
    Transform the h5ad files for several species into the torch tensors
    """

    def __init__(self, species_to_adata:dict,species_celltype_labels:dict,species_batch_labels:dict):

        super(SingleCellDataset,self).__init__()
        self.raw_counts = {}
        self.species_num = len(species_to_adata)
        self.species_category = list(species_to_adata.keys())
        self.num_cells ={}
        self.num_genes = {}
        self.celltype_labels = {}  # this is the cell type labels for each species, length is the number of cells
        self.batch_labels = {} # this is the batch labels for each species, length is the number of cells
        self.species_labels = {} # this is the species labels for each species, length is the number of cells
        self.celltype_category = {} # this is the unique cell type category for each species
        self.celltype_onehot_embedding = {} 
        self.celltype_id_embedding = {}

        # load the count data and the cell type label
        for species_label, file in species_to_adata.items():
            adata = file
            #import pdb;pdb.set_trace()
            X = torch.Tensor(adata.X)
            num_cells,num_genes = X.shape
            self.raw_counts[species_label] = torch.Tensor(X)
            self.num_cells[species_label] = num_cells
            self.num_genes[species_label] = num_genes
            self.celltype_labels[species_label] = adata.obs[species_celltype_labels[species_label]].values
            self.species_labels[species_label] = [species_label]*num_cells
            if species_batch_labels is not None:
                self.batch_labels[species_label] = adata.obs[species_batch_labels[species_label]].values
            # this is to get the batch label in the species level (mainly for nested batch effect)
            self.celltype_category[species_label]=list(np.unique(self.celltype_labels[species_label]))
    
        species_labels = self.species_category
        self.species_id_dict = dict(zip(species_labels, range(len(species_labels))))  
        onehot_species_id = np.eye(len(species_labels), dtype=np.float32)
        species_onehot_dict = dict(zip(species_labels, onehot_species_id))
        self.species_onehot_embedding = {species: np.array([species_onehot_dict[k] for k in self.species_labels[species]]) for species in species_labels}
        self.species_id_embedding = {species: np.array([self.species_id_dict[k] for k in self.species_labels[species]]) for species in species_labels} # this is the index code for the species

        self.all_unique_celltype = set(val for key in self.celltype_category.keys() for val in self.celltype_category[key])
        #np.unique(self.celltype_category.values())
        self.cell_type_num = len(self.all_unique_celltype)
        self.cell_type_id_dict = dict(zip(self.all_unique_celltype, range(len(self.all_unique_celltype))))
        onehot_celltype_id = np.eye(self.cell_type_num, dtype=np.float32)
        celltype_onehot_dict = dict(zip(self.all_unique_celltype, onehot_celltype_id))
        self.celltype_onehot_embedding = {species: np.array([celltype_onehot_dict[k] for k in self.celltype_labels[species]]) for species in species_labels}
        self.celltype_id_embedding = {species: np.array([self.cell_type_id_dict[k] for k in self.celltype_labels[species]]) for species in species_labels}
        self.trans_profiles = torch.vstack(list(self.raw_counts.values()))
        self.species_onehot_embedding_all = np.vstack(list(self.species_onehot_embedding.values()))
        self.species_id_embedding_all = np.concatenate(list(self.species_id_embedding.values()))
        self.celltype_onehot_embedding_all = np.vstack(list(self.celltype_onehot_embedding.values()))
        self.celltype_id_embedding_all = np.concatenate(list(self.celltype_id_embedding.values()))

    def __len__(self):
        return sum(self.num_cells.values())
    
    def get_dim(self):
        return self.num_genes

    def __getitem__(self,idx):
        return self.trans_profiles[idx], self.species_onehot_embedding_all[idx], self.species_id_embedding_all[idx], self.celltype_onehot_embedding_all[idx], self.celltype_id_embedding_all[idx]

def multi_species_collate_fn(batch):
    '''this function is to concatenate the data from different species into a single batch'''
    species_to_label=defaultdict(list) # this is to store the data from different species
    species_to_rawcount=defaultdict(list) # this is to store the label from different species
    species_to_onehot_embedding=defaultdict(list)
    species_to_id_embedding=defaultdict(list)
    species_to_celltype_onehot_embedding=defaultdict(list)
    species_to_celltype_id_embedding=defaultdict(list)
    species_to_batch=defaultdict(list)
    for species, raw_count, onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch in batch:
        species_to_label[species].append(species)
        species_to_rawcount[species].append(raw_count)
        species_to_onehot_embedding[species].append(onehot_embedding)
        species_to_id_embedding[species].append(id_embedding)
        species_to_celltype_onehot_embedding[species].append(celltype_onehot_embedding)
        species_to_celltype_id_embedding[species].append(celltype_id_embedding)
        has_batch_labels = False

        if batch is not None:
            species_to_batch[species].append(batch)
            has_batch_labels = True
    
    # assert 1 <= len(species_to_label) <= 2, "Only support 1 or 2 species"
    batch_dict = {}
    all_species = sorted(list(species_to_label.keys()))
    for species in all_species:
        if has_batch_labels:
            raw_count, onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch = torch.stack(species_to_rawcount[species]), torch.tensor(np.stack(species_to_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_id_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_id_embedding[species],axis=0)), torch.tensor(np.stack(species_to_batch[species],axis=0))
        else:
            raw_count, onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch = torch.stack(species_to_rawcount[species]), torch.tensor(np.stack(species_to_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_id_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_id_embedding[species],axis=0)), None
            
        batch_dict[species] = (raw_count, onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch)
    return batch_dict #{species:[raw_count, onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch]}


def multi_species_collate_fn_esm_llama(batch):
    '''this function is to concatenate the data from different species into a single batch'''
    species_to_label=defaultdict(list) # this is to store the data from different species
    species_to_rawcount=defaultdict(list) # this is to store the label from different species
    species_to_rawcount_esm=defaultdict(list)
    species_to_rawcount_llama=defaultdict(list)
    species_to_onehot_embedding=defaultdict(list)
    species_to_id_embedding=defaultdict(list)
    species_to_celltype_onehot_embedding=defaultdict(list)
    species_to_celltype_id_embedding=defaultdict(list)
    species_to_batch=defaultdict(list)
    for species, raw_count, raw_count_esm, raw_count_llama,onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch in batch:
        species_to_label[species].append(species)
        species_to_rawcount[species].append(raw_count)
        species_to_rawcount_esm[species].append(raw_count_esm)
        species_to_rawcount_llama[species].append(raw_count_llama)
        species_to_onehot_embedding[species].append(onehot_embedding)
        species_to_id_embedding[species].append(id_embedding)
        species_to_celltype_onehot_embedding[species].append(celltype_onehot_embedding)
        species_to_celltype_id_embedding[species].append(celltype_id_embedding)
        has_batch_labels = False

        if batch is not None:
            species_to_batch[species].append(batch)
            has_batch_labels = True
    
    # assert 1 <= len(species_to_label) <= 2, "Only support 1 or 2 species"
    batch_dict = {}
    all_species = sorted(list(species_to_label.keys()))
    for species in all_species:
        if has_batch_labels:
            raw_count, raw_count_esm,raw_count_llama,onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch = torch.stack(species_to_rawcount[species]), torch.stack(species_to_rawcount_esm[species]),torch.stack(species_to_rawcount_llama[species]),torch.tensor(np.stack(species_to_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_id_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_id_embedding[species],axis=0)), torch.tensor(np.stack(species_to_batch[species],axis=0))
        else:
            raw_count, raw_count_esm,raw_count_llama,onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch = torch.stack(species_to_rawcount[species]), torch.stack(species_to_rawcount_esm[species]),torch.stack(species_to_rawcount_llama[species]),torch.tensor(np.stack(species_to_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_id_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_onehot_embedding[species],axis=0)), torch.tensor(np.stack(species_to_celltype_id_embedding[species],axis=0)), None
            
        batch_dict[species] = (raw_count, raw_count_esm,raw_count_llama,onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch)
    return batch_dict #{species:[raw_count, onehot_embedding, id_embedding, celltype_onehot_embedding, celltype_id_embedding, batch]}


class SingleCellDataset_finetune(Dataset):
    """
    for the finetune process, take the macrogene expression as input, which is the results of raw count * pretrained macrogene weights
    """

    def __init__(self, macrogene_exp,species_labels,celltype_labels):
        #import pdb;pdb.set_trace()
        self.trans_profiles = torch.Tensor(macrogene_exp)   # shape is cells * genes
        self.gene_num = self.trans_profiles.shape[1]
        self.cell_type_unique = np.unique(celltype_labels)
        self.cell_type_num = len(self.cell_type_unique)
        self.species_labels_unique=np.unique(species_labels)
        self.species_num = len(self.species_labels_unique)
        self.species_id = species_labels
        self.species_id_dict = dict(zip(self.species_labels_unique, range(len(self.species_labels_unique))))  
        onehot_species_id = np.eye(self.species_num, dtype=np.float32)
        species_onehot_dict = dict(zip(self.species_labels_unique, onehot_species_id))
        self.species_category = np.unique(species_labels)
        self.species_onehot_embedding = np.array([species_onehot_dict[k] for k in self.species_id]) # this is the onehot encoding for the species
        self.species_id_embedding = np.array([self.species_id_dict[k] for k in self.species_id])# this is the index code for the species
        onehot_celltype_id = np.eye(self.cell_type_num, dtype=np.float32)
        celltype_onehot_dict = dict(zip(self.cell_type_unique, onehot_celltype_id))
        self.celltype_category = np.unique(celltype_labels)
        self.celltype_category = np.ravel(self.celltype_category)
        self.cell_type_id_dict = dict(zip(self.cell_type_unique, range(len(self.cell_type_unique))))
        #import pdb;pdb.set_trace()
        self.celltype_onehot_embedding = np.array([celltype_onehot_dict[k] for k in celltype_labels])
        self.celltype_id_embedding = np.array([self.cell_type_id_dict[k] for k in celltype_labels])


    def __len__(self):
        return len(self.trans_profiles)
    
    def __getitem__(self,idx):
        return self.trans_profiles[idx], self.species_onehot_embedding[idx], self.species_id_embedding[idx], self.celltype_onehot_embedding[idx], self.celltype_id_embedding[idx]
    #genes, species, species_idx, degs


class SingleCellDataset_esm_llama(Dataset):
    """
    Transform the h5ad files for several species into the torch tensors
    """

    def __init__(self, species_to_adata:dict,species_to_adata_esm:dict,species_to_adata_llama:dict,species_celltype_labels:dict,species_batch_labels:dict):
        # species_to_adata: the origin h5ad files for the raw count data
        # species_to_adata_esm: the h5ad files for the esm filtered adata
        # species_to_adata_llama: the h5ad files for the llama filtered adata

        super(SingleCellDataset_esm_llama,self).__init__()
        self.raw_counts = {}
        self.species_num = len(species_to_adata)
        self.species_category = list(species_to_adata.keys())
        self.num_cells ={}
        self.num_genes = {}
        self.celltype_labels = {}  # this is the cell type labels for each species, length is the number of cells
        self.batch_labels = {} # this is the batch labels for each species, length is the number of cells
        self.species_labels = {} # this is the species labels for each species, length is the number of cells
        self.celltype_category = {} # this is the unique cell type category for each species
        self.celltype_onehot_embedding = {} 
        self.celltype_id_embedding = {}

        # load the count data and the cell type label
        for species_label, file in species_to_adata.items():
            adata = file
            #import pdb;pdb.set_trace()
            X = torch.Tensor(adata.X)
            num_cells,num_genes = X.shape
            self.raw_counts[species_label] = torch.Tensor(X)
            self.num_cells[species_label] = num_cells
            self.num_genes[species_label] = num_genes
            self.celltype_labels[species_label] = adata.obs[species_celltype_labels[species_label]].values
            self.species_labels[species_label] = [species_label]*num_cells
            if species_batch_labels is not None:
                self.batch_labels[species_label] = adata.obs[species_batch_labels[species_label]].values
            # this is to get the batch label in the species level (mainly for nested batch effect)
            self.celltype_category[species_label]=list(np.unique(self.celltype_labels[species_label]))
        
        # load the esm data
        for species_label, file in species_to_adata_esm.items():
            adata = file
            #import pdb;pdb.set_trace()
            X = torch.Tensor(adata.X)
            num_cells,num_genes = X.shape
            self.raw_counts[species_label+'_esm'] = torch.Tensor(X)
            self.num_genes[species_label+'_esm'] = num_genes

        # load the llama data
        for species_label, file in species_to_adata_llama.items():
            adata = file
            #import pdb;pdb.set_trace()
            X = torch.Tensor(adata.X)
            num_cells,num_genes = X.shape
            self.raw_counts[species_label+'_llama'] = torch.Tensor(X)
            self.num_genes[species_label+'_llama'] = num_genes
    
        species_labels = self.species_category
        self.species_id_dict = dict(zip(species_labels, range(len(species_labels))))  
        onehot_species_id = np.eye(len(species_labels), dtype=np.float32)
        species_onehot_dict = dict(zip(species_labels, onehot_species_id))
        self.species_onehot_embedding = {species: np.array([species_onehot_dict[k] for k in self.species_labels[species]]) for species in species_labels}
        self.species_id_embedding = {species: np.array([self.species_id_dict[k] for k in self.species_labels[species]]) for species in species_labels} # this is the index code for the species

        self.all_unique_celltype = set(val for key in self.celltype_category.keys() for val in self.celltype_category[key])
        #np.unique(self.celltype_category.values())
        self.cell_type_num = len(self.all_unique_celltype)
        self.cell_type_id_dict = dict(zip(self.all_unique_celltype, range(len(self.all_unique_celltype))))
        onehot_celltype_id = np.eye(self.cell_type_num, dtype=np.float32)
        celltype_onehot_dict = dict(zip(self.all_unique_celltype, onehot_celltype_id))
        self.celltype_onehot_embedding = {species: np.array([celltype_onehot_dict[k] for k in self.celltype_labels[species]]) for species in species_labels}
        self.celltype_id_embedding = {species: np.array([self.cell_type_id_dict[k] for k in self.celltype_labels[species]]) for species in species_labels}

    def __len__(self):
        return sum(self.num_cells.values())
    
    def get_dim(self):
        return self.num_genes

    def __getitem__(self,idx):
        if isinstance(idx, int):
            count = 0
            for species in self.species_category:
                if idx < self.num_cells[species]:
                    if len(self.batch_labels) != 0:
                        batch_ret = self.batch_labels[species][idx]
                    else:
                        batch_ret = None
                    return species, self.raw_counts[species][idx], self.raw_counts[species+'_esm'][idx],self.raw_counts[species+'_llama'][idx],self.species_onehot_embedding[species][idx], self.species_id_embedding[species][idx], self.celltype_onehot_embedding[species][idx], self.celltype_id_embedding[species][idx], batch_ret
                else:
                    idx -= self.num_cells[species]
            raise IndexError
        else:
            raise NotImplementedError




class SingleCellDataset_for_raw_reconstruction(Dataset):
    """
    This is the dataset object for taking raw count data as input.
    It processes both ESM and Llama gene expression datasets,
    creates a universal gene set, calculates macrogenes, and handles metadata.
    """

    def __init__(self, species_to_adata_esm: dict, species_to_adata_llama: dict,
                 esm_centroid_weights: dict, llama_centroid_weights: dict,
                 species_celltype_labels: dict, species_batch_labels: dict = None):
        """
        Initializes the dataset.

        Args:
            species_to_adata_esm (dict): Dictionary mapping species name to AnnData object for ESM raw counts.
            species_to_adata_llama (dict): Dictionary mapping species name to AnnData object for Llama raw counts.
            esm_centroid_weights (dict): Dictionary mapping species name to ESM centroid weights (Tensor).
                                          Expected shape: (num_genes_esm, num_centroids_esm).
            llama_centroid_weights (dict): Dictionary mapping species name to Llama centroid weights (Tensor).
                                           Expected shape: (num_genes_llama, num_centroids_llama).
            species_celltype_labels (dict): Dictionary mapping species name to the column name for cell type labels.
            species_batch_labels (dict, optional): Dictionary mapping species name to the column name for batch labels. Defaults to None.
        """

        super(SingleCellDataset_for_raw_reconstruction, self).__init__()

        self.raw_counts_esm = {}
        self.raw_counts_llama = {}
        self.raw_counts_universe_per_species = {}  # Stores the union of ESM and Llama genes for each species, with species prefix

        # Macrogenes calculated per model and stored separately for flexible retrieval
        self.macrogenes_esm = {}
        self.macrogenes_llama = {}
        self.macrogenes_combined_per_species = {} # Stores concatenated macrogenes for each species

        self.species_category = list(species_to_adata_esm.keys())
        if set(self.species_category) != set(species_to_adata_llama.keys()):
            raise ValueError("Species keys in species_to_adata_esm and species_to_adata_llama must match.")

        self.num_cells = {}
        self.num_genes_universe_per_species = {}  # Number of genes in the universe set for each species

        # Temporary dictionaries to hold metadata per species before global stacking
        temp_celltype_labels = {}
        temp_batch_labels = {}
        temp_species_labels = {}
        temp_celltype_category = {} # This is not directly stacked, but derived from the temporary celltype_labels

        # New: Store global indices per species for custom sampling
        # (This is still useful for internal mapping even if __getitem__ doesn't use it directly for lookup)
        self.species_indices = defaultdict(list)
        global_idx_counter = 0

        # For global universal gene names
        all_species_prefixed_genes = set()
        self.gene_names_universe = {} # Stores the species-prefixed gene names for each species' universe

        self.global_species_ids_for_weights = []
        # --- Load and Process Data for ESM and Llama for each species ---
        for species_label in self.species_category:
            # Load ESM data
            adata_esm = species_to_adata_esm[species_label]
            # Convert sparse to dense if necessary
            X_esm = torch.Tensor(adata_esm.X.toarray() if hasattr(adata_esm.X, 'toarray') else adata_esm.X).to(torch.float64)
            self.raw_counts_esm[species_label] = X_esm
            num_cells_esm, num_genes_esm = X_esm.shape

            # Load Llama data
            adata_llama = species_to_adata_llama[species_label]
            # Convert sparse to dense if necessary
            X_llama = torch.Tensor(adata_llama.X.toarray() if hasattr(adata_llama.X, 'toarray') else adata_llama.X).to(torch.float64)
            self.raw_counts_llama[species_label] = X_llama
            num_cells_llama, num_genes_llama = X_llama.shape

            # Ensure consistent number of cells between ESM and Llama for the same species
            if num_cells_esm != num_cells_llama:
                raise ValueError(
                    f"Mismatch in number of cells for species {species_label} "
                    f"between ESM ({num_cells_esm}) and Llama ({num_cells_llama})."
                )

            self.num_cells[species_label] = num_cells_esm

            # Populate species_indices for sampling
            self.species_indices[species_label] = list(range(global_idx_counter, global_idx_counter + num_cells_esm))

            species_id_val = self.species_id_dict[species_label] if hasattr(self, 'species_id_dict') else 0 # Will be updated later
            self.global_species_ids_for_weights.extend([species_label] * num_cells_esm) # Store actual species label for now

            global_idx_counter += num_cells_esm

            # --- Handle Metadata (assuming consistent across ESM and Llama for same species) ---
            temp_celltype_labels[species_label] = adata_esm.obs[species_celltype_labels[species_label]].values
            temp_species_labels[species_label] = np.array([species_label] * num_cells_esm, dtype=object) # Store as object to avoid numpy type issues

            if species_batch_labels is not None and species_label in species_batch_labels:
                # Ensure batch labels are always treated as objects/strings if they contain None
                temp_batch_labels[species_label] = adata_esm.obs[species_batch_labels[species_label]].astype(object).values
            else:
                temp_batch_labels[species_label] = np.array([None] * num_cells_esm, dtype=object) # Initialize with None if no batch labels
            temp_celltype_category[species_label] = list(np.unique(temp_celltype_labels[species_label]))

            # --- Create Universal Raw Counts for this species with species prefix ---
            genes_esm = set(adata_esm.var_names.tolist())
            genes_llama = set(adata_llama.var_names.tolist())
            
            # Add species prefix to gene names
            species_prefixed_genes_esm = {f"{species_label}_{gene}" for gene in genes_esm}
            species_prefixed_genes_llama = {f"{species_label}_{gene}" for gene in genes_llama}
            universe_genes_for_species = sorted(list(species_prefixed_genes_esm.union(species_prefixed_genes_llama)))
            self.num_genes_universe_per_species[species_label] = len(universe_genes_for_species)
            self.gene_names_universe[species_label] = universe_genes_for_species # Store for this species

            # Add to the global set of all unique genes
            all_species_prefixed_genes.update(universe_genes_for_species)

            # Create mappings for gene indices
            esm_gene_to_idx = {gene: i for i, gene in enumerate(adata_esm.var_names)}
            llama_gene_to_idx = {gene: i for i, gene in enumerate(adata_llama.var_names)}
            universe_gene_to_idx_for_species = {gene: i for i, gene in enumerate(universe_genes_for_species)}

            # Initialize universal raw count matrix for this species
            universe_raw_counts_matrix = torch.zeros((num_cells_esm, len(universe_genes_for_species)), dtype=X_esm.dtype)

            # Populate with ESM data
            for gene in genes_esm:
                species_prefixed_gene = f"{species_label}_{gene}"
                universe_idx = universe_gene_to_idx_for_species[species_prefixed_gene]
                esm_idx = esm_gene_to_idx[gene]
                universe_raw_counts_matrix[:, universe_idx] += X_esm[:, esm_idx]

            # Get non-overlapping genes only (i.e., those not in ESM)
            non_overlapping_genes_llama = genes_llama - genes_esm

            # Populate only non-overlapping genes from Llama
            for gene in non_overlapping_genes_llama:
                species_prefixed_gene = f"{species_label}_{gene}"
                universe_idx = universe_gene_to_idx_for_species[species_prefixed_gene]
                llama_idx = llama_gene_to_idx[gene]
                universe_raw_counts_matrix[:, universe_idx] = X_llama[:, llama_idx]


            self.raw_counts_universe_per_species[species_label] = universe_raw_counts_matrix

            # --- Calculate Macrogenes ---
            esm_weights = torch.stack(esm_centroid_weights[species_label])
            llama_weights = torch.stack(llama_centroid_weights[species_label])
            
            # Verify weight dimensions
            if esm_weights.shape[0] != num_genes_esm:
                raise ValueError(
                    f"ESM centroid weights for {species_label} ({esm_weights.shape}) "
                    f"do not match gene dimension of raw counts ({num_genes_esm}). "
                    f"Expected shape: (num_genes_esm, num_centroids_esm)."
                )
            if llama_weights.shape[0] != num_genes_llama:
                raise ValueError(
                    f"Llama centroid weights for {species_label} ({llama_weights.shape}) "
                    f"do not match gene dimension of raw counts ({num_genes_llama}). "
                    f"Expected shape: (num_genes_llama, num_centroids_llama)."
                )

            # Perform matrix multiplication
            esm_macrogenes = X_esm @ esm_weights
            llama_macrogenes = X_llama @ llama_weights

            self.macrogenes_esm[species_label] = esm_macrogenes
            self.macrogenes_llama[species_label] = llama_macrogenes

            # Combine ESM and Llama macrogenes for this species
            self.macrogenes_combined_per_species[species_label] = torch.cat((esm_macrogenes, llama_macrogenes), dim=1)

        # --- Global Metadata Processing ---
        # Collect all unique cell types across all species
        # Note: Using temp_celltype_category for this initial aggregation
        self.all_unique_celltype = set(val for key in temp_celltype_category.keys() for val in temp_celltype_category[key])
        self.cell_type_num = len(self.all_unique_celltype)
        self.cell_type_id_dict = dict(zip(sorted(list(self.all_unique_celltype)), range(len(self.all_unique_celltype))))

        # Create one-hot and ID embeddings for cell types and species, and stack globally
        # These will be the actual attributes used by __getitem__
        self.global_celltype_onehot_embedding = []
        self.global_celltype_id_embedding = []
        self.global_species_onehot_embedding = []
        self.global_species_id_embedding = []
        self.global_batch_labels = [] # List of objects (strings or None)

        species_labels_unique = sorted(self.species_category)
        self.species_id_dict = dict(zip(species_labels_unique, range(len(species_labels_unique))))
        onehot_species_id_matrix = np.eye(len(species_labels_unique), dtype=np.float32)
        
        onehot_celltype_id_matrix = np.eye(self.cell_type_num, dtype=np.float32)

        for species_label in self.species_category:
            # Species embeddings
            species_onehot_for_species = onehot_species_id_matrix[self.species_id_dict[species_label]]
            self.global_species_onehot_embedding.extend([species_onehot_for_species] * self.num_cells[species_label])
            self.global_species_id_embedding.extend([self.species_id_dict[species_label]] * self.num_cells[species_label])

            # Cell type embeddings
            celltype_labels_for_species = temp_celltype_labels[species_label]
            for celltype in celltype_labels_for_species:
                self.global_celltype_onehot_embedding.append(onehot_celltype_id_matrix[self.cell_type_id_dict[celltype]])
                self.global_celltype_id_embedding.append(self.cell_type_id_dict[celltype])
            
            # Batch labels
            self.global_batch_labels.extend(temp_batch_labels[species_label].tolist()) # Convert numpy array to list for consistent type with None

        # Convert lists of arrays to stacked numpy arrays for efficient indexing
        self.global_celltype_onehot_embedding = np.array(self.global_celltype_onehot_embedding, dtype=np.float32)
        self.global_celltype_id_embedding = np.array(self.global_celltype_id_embedding, dtype=np.int64)
        self.global_species_onehot_embedding = np.array(self.global_species_onehot_embedding, dtype=np.float32)
        self.global_species_id_embedding = np.array(self.global_species_id_embedding, dtype=np.int64)
        # self.global_batch_labels remains a list of objects


        # --- Final Stacking of Macrogenes and Universal Raw Counts ---
        self.macrogenes_combined = torch.vstack(list(self.macrogenes_combined_per_species.values()))

        self.all_universe_genes_sorted = sorted(list(all_species_prefixed_genes)) # The global ordered list of all genes
        global_gene_to_idx = {gene: i for i, gene in enumerate(self.all_universe_genes_sorted)}

        total_cells_all_species = sum(self.num_cells.values())
        total_unique_genes = len(self.all_universe_genes_sorted)

        self.universal_raw_counts_stacked = torch.zeros((total_cells_all_species, total_unique_genes), dtype=torch.float64)

        current_cell_row_offset = 0
        for species_label in self.species_category:
            species_raw_counts_matrix = self.raw_counts_universe_per_species[species_label]
            species_prefixed_gene_names = self.gene_names_universe[species_label]
            num_cells_in_species = self.num_cells[species_label]

            for local_gene_idx, species_prefixed_gene in enumerate(species_prefixed_gene_names):
                global_gene_idx = global_gene_to_idx[species_prefixed_gene]
                self.universal_raw_counts_stacked[current_cell_row_offset : current_cell_row_offset + num_cells_in_species, global_gene_idx] = \
                    species_raw_counts_matrix[:, local_gene_idx]
            current_cell_row_offset += num_cells_in_species

        # *** NEW: Calculate and store sample weights ***
        self._calculate_sample_weights()

    def _calculate_sample_weights(self):
        """Calculates weights for each sample to balance species sampling."""
        
        # Get counts for each species
        species_counts = defaultdict(int)
        for species_label in self.global_species_ids_for_weights:
            species_counts[species_label] += 1
        
        # Find the maximum count among species
        max_count = max(species_counts.values())

        # Calculate inverse frequency weights
        # Assign a higher weight to samples from less frequent species
        species_weights = {species: max_count / count for species, count in species_counts.items()}
        
        # Create a list of weights for each individual sample
        self.sample_weights = torch.tensor([species_weights[self.global_species_ids_for_weights[i]] 
                                            for i in range(len(self))])

    def get_stacked_macrogenes(self):
        """
        Returns a tensor of macrogenes across all species,
        where each cell has ESM and Llama macrogenes concatenated.

        Returns:
            torch.Tensor: Shape [total_cells, esm_dim + llama_dim]
        """
        return self.macrogenes_combined
    
    def get_stacked_universal_raw_counts(self):
        """
        Returns a stacked raw count matrix across all species.
        For genes not present in a species, fills with zeros.

        Returns:
            torch.Tensor: Shape [total_cells, total_genes_union]
            List[str]: Ordered list of all unique genes (columns of the matrix)
        """
        return self.universal_raw_counts_stacked, self.all_universe_genes_sorted
    
        
    def __len__(self):
        """Returns the total number of cells across all species."""
        return sum(self.num_cells.values())

    def get_dim(self):
        """Returns a dictionary of the number of genes in the universal set for each species.
        Note: This now refers to the 'universal' gene set *for each species*, not the global universal set."""
        return self.num_genes_universe_per_species

    def __getitem__(self, idx: int):
        """
        Retrieves a single cell's data based on a global index.

        Args:
            idx (int): The global index of the cell.

        Returns:
            tuple: A tuple containing:
                - universal_raw_counts (torch.Tensor): Raw counts for the cell aligned to the global universal gene set.
                                                        Shape: (total_unique_genes,).
                - macrogenes_combined (torch.Tensor): Combined (ESM and Llama) macrogenes for the cell.
                                                    Shape: (num_centroids_esm + num_centroids_llama,).
                - species_onehot_embedding (np.ndarray): One-hot encoding of the species.
                - species_id_embedding (np.ndarray): Integer ID of the species.
                - celltype_onehot_embedding (np.ndarray): One-hot encoding of the cell type.
                - celltype_id_embedding (np.ndarray): Integer ID of the cell type.
                - batch_ret (Any): Batch label of the cell, or None if not provided.
        """
        if not isinstance(idx, int):
            raise NotImplementedError("Only integer indexing is supported.")

        # Directly retrieve the relevant data using the global index 'idx'
        #import pdb;pdb.set_trace()
        universal_raw_counts_for_cell = self.universal_raw_counts_stacked[idx]
        macrogenes_combined_for_cell = self.macrogenes_combined[idx]

        species_onehot = self.global_species_onehot_embedding[idx]
        species_id = self.global_species_id_embedding[idx]
        celltype_onehot = self.global_celltype_onehot_embedding[idx]
        celltype_id = self.global_celltype_id_embedding[idx]
        batch_ret = self.global_batch_labels[idx]

        return (universal_raw_counts_for_cell,
                macrogenes_combined_for_cell,
                species_onehot,
                species_id,
                celltype_onehot,
                celltype_id,
                batch_ret)

if __name__ == "__main__":
    species_h5ad_files={'cat':'../data/task3_cat.h5ad','tiger':'../data/task3_tiger.h5ad'}
    species_celltype_labels={'cat':'NewCelltype','tiger':'NewCelltype'}
    species_batch_labels=None
    dataset=SingleCellDataset(species_to_adata=species_h5ad_files,species_celltype_labels=species_celltype_labels,species_batch_labels=species_batch_labels)
    import pdb;pdb.set_trace()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True,collate_fn=multi_species_collate_fn)
    for i, batch in enumerate(dataloader):
        print(batch)
        break
