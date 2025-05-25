import torch 
import torch.nn as nn
import numpy as np
import wandb
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import pandas as pd

## below code, i am going to add a module like saturn
# first for the generated protein embeddings from the esm model, i will need to initialize the macrogenes using k-mean clustering
# then i need to initialize the weight for each gene to each macrogenes
def macrogene_initialization(prot_embeddings, species_gene_names, num_macrogene=2000, normalize=False,seed=0):
    print('macrogene initialization')
    if normalize:
        row_sums = prot_embeddings.sum(axis=1)
        prot_embeddings = prot_embeddings / row_sums[:, np.newaxis]
    
    kmeans_obj = KMeans(n_clusters=num_macrogene, random_state=seed).fit(prot_embeddings)
    # get the distance from each gene to the macrogene
    dist = kmeans_obj.transform(prot_embeddings)
    # get the score for each gene
    score = default_score_function(dist)

    species_gene_scores = {}
    for i, gene in enumerate(species_gene_names):
        species_gene_scores[gene] = score[i,:]
    return species_gene_scores


def default_score_function(dist):
    '''
    default score function for the distance from the gene to the macrogene
    '''
    ranked = rankdata(dist, axis=1) # ranking the genes to the macrogene distance, 1 is the closest

    to_score = np.log1p(1/ranked)
    to_score = ((to_score)**2) * 2
    return to_score

# next step is to updata all the macrogene weights for each gene
def update_macrogene_weight(species_to_adata,sorted_species_names, species_gene_scores, num_macrogene=2000, num_gene=10000):
    print('update macrogene weight')
    macrogene_weights = []
    all_species_gene_names = []
    for species in sorted_species_names:
        adata = species_to_adata[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        species_gene_names = list(species_str.str.cat(gene_names,sep='_'))
        all_species_gene_names = all_species_gene_names + species_gene_names
        
        for sgn in species_gene_names:
            macrogene_weights.append(torch.tensor(species_gene_scores[sgn]))
    macrogene_weights = torch.stack(macrogene_weights)

    return macrogene_weights, all_species_gene_names
'''
def generate_macrogene_input(adata, species_gene_names, macrogene_weights, num_macrogene=2000):
    print('generate macrogene input')
    gene_expression = adata.X
    gene_expression = torch.tensor(gene_expression.toarray())
    macrogene_input = torch.zeros((gene_expression.shape[0], num_macrogene))
    for i, gene in enumerate(species_gene_names):
        macrogene_input[:,i] = gene_expression[:,i] * macrogene_weights[i]
    return macrogene_input
'''
def load_gene_embeddings_adata(adata, species, embedding_path):
    '''
    load the gene embeddings from the given embedding path.
    :param adata: the adata object
    :param species: the species name, could be a list of species name
    :param embedding_path: the path to the gene embeddings

    :return: A tuple containing:
        - a subset of the adata object only containing the gene expression for genes with embedding
        - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim)
    '''
    # get species name
    species_names = species
    species_names_set = set(species_names)

    # load the gene embeddings
    # make a dictionary to store as {species: {gene_symbol: embedding}}
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower():gene_embedding
            for gene_symbol, gene_embedding in torch.load(embedding_path).items()
        }
        for species in species_names
    }

    # get the gene names that are in the embedding
    genes_with_embeddings = set.intersection(*[
        set(gene_symbol_to_embedding)
        for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()
    ])
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}

    # subset adata based on the genes with embeddings
    adata_subset = adata[:, adata.var_names.isin(genes_to_use)]

    # make a dictionary to store as {species: gene_embedding_matrix}
    species_to_gene_embedding_matrix = {
        species_name: torch.stack([
            species_to_gene_symbol_to_embedding[species_name][gene.lower()]
            for gene in adata_subset.var_names
        ])
        for species_name in species_names
    }

    return adata_subset, species_to_gene_embedding_matrix


def load_gene_embeddings_adata_esm_llama(adata, species, embedding_path1, embedding_path2):
    '''
    Load the gene embeddings from the given embedding paths.
    :param adata: the adata object
    :param species: the species name, could be a list of species names
    :param embedding_path1: the first path to the gene embeddings
    :param embedding_path2: the second path to the gene embeddings

    :return: A tuple containing:
        - a subset of the adata object only containing the gene expression for genes with embeddings in both paths
        - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim)
    '''
    # Get species name
    species_names = species
    species_names_set = set(species_names)

    # Load the gene embeddings from the first path
    species_to_gene_symbol_to_embedding1 = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(embedding_path1).items()
        }
        for species in species_names
    }

    # Load the gene embeddings from the second path
    species_to_gene_symbol_to_embedding2 = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(embedding_path2).items()
        }
        for species in species_names
    }

    # Get the gene names that are in both embeddings
    genes_with_embeddings1 = set.intersection(*[
        set(gene_symbol_to_embedding)
        for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding1.values()
    ])
    genes_with_embeddings2 = set.intersection(*[
        set(gene_symbol_to_embedding)
        for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding2.values()
    ])
    common_genes_with_embeddings = genes_with_embeddings1.intersection(genes_with_embeddings2)

    # Subset adata based on the common genes with embeddings
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in common_genes_with_embeddings}
    adata_subset = adata[:, adata.var_names.isin(genes_to_use)]

    # Make a dictionary to store {species: gene_embedding_matrix} based on the common genes
    species_to_gene_embedding_matrix1 = {
        species_name: torch.stack([
            species_to_gene_symbol_to_embedding1[species_name][gene.lower()]
            for gene in adata_subset.var_names
        ])
        for species_name in species_names
    }

    species_to_gene_embedding_matrix2 = {
        species_name: torch.stack([
            species_to_gene_symbol_to_embedding2[species_name][gene.lower()]
            for gene in adata_subset.var_names
        ])
        for species_name in species_names
    }

    return adata_subset, species_to_gene_embedding_matrix1, species_to_gene_embedding_matrix2
