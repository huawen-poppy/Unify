import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch.nn as nn
import anndata as ann
from collections import defaultdict

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
