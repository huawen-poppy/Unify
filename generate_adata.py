from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from models import Decoder, Discriminator, Discriminator_celltype, Encoder
import wandb
from macrogene_initialize import macrogene_initialization,load_gene_embeddings_adata
import anndata as ad
import pandas as pd
import numpy as np
import torch
from dataset import SingleCellDataset,multi_species_collate_fn
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import scib
import argparse
import yaml
import os
import torch
from min_norm_solvers import MinNormSolver, gradient_normalizers
from tqdm import tqdm
import copy
from anndata import AnnData 

def main():
    parser = argparse.ArgumentParser(description='Train scAAE model with species and celltype classifiers.')
    parser.add_argument('--output_model_path', type=str, help='Path to save the trained model,specifically is a folder name',default='../models/aae_species_celltype/task3/')
    parser.add_argument('--input_model_folder',type=str,help='Path to the input model folder',default='../models/aae_species_celltype/task3/')

    parser.add_argument('--encoder_hidden_size', type=int, help='Hidden size for the encoder',default=128)
    parser.add_argument('--decoder_hidden_size', type=int, help='Hidden size for the decoder',default=128)
    parser.add_argument('--latent_size', type=int, help='Latent size for the encoder and decoder',default=10)
    parser.add_argument('--encoder_dropout_ratio', type=float, help='Dropout ratio for the encoder',default=0.2)
    parser.add_argument('--decoder_dropout_ratio', type=float, help='Dropout ratio for the decoder',default=0.2)
    parser.add_argument('--used_epoch', type=int, help='Used epoch for the model',default=394)
    parser.add_argument('--task_name', type=str, help='Name of the task',default='task3')
    parser.add_argument('--species_batch_labels', type=str, nargs='+',help='Species batch labels for the input data files, should be a list of strings',default=None)
    
    parser.add_argument('--h5ad_files', nargs='+', help='list of h5ad files')
    parser.add_argument('--species_labels', nargs='+', help='list of species labels')
    parser.add_argument('--celltype_labels', nargs='+', help='list of celltype labels')
    parser.add_argument('--gene_esm_embedding_path', nargs='+', help='list of gene esm embedding path')
    parser.add_argument('--gene_llama_embedding_path', nargs='+', help='list of gene llama embedding path')
    parser.add_argument('--num_esm_macrogene', type=int, help='number of esm macrogene',default=2000)
    parser.add_argument('--num_llama_macrogene', type=int, help='number of llama macrogene',default=2000)
    parser.add_argument('--seed', type=int, help='random seed',default=0)
    parser.add_argument('--output_path', type=str, help='output path for saving')
    parser.add_argument('--highly_variable_genes', type=int, help='number of highly variable genes',default=8000)
    parser.add_argument('--batch_labels', type=str, help='nested batch labels',default=None)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## in below code, I am trying to train the supervised aae with an extra macrogene layer
    h5ad_files=args.h5ad_files
    species_labels=args.species_labels
    celltype_labels=args.celltype_labels
    gene_esm_embedding_path=args.gene_esm_embedding_path
    gene_llama_embedding_path=args.gene_llama_embedding_path
    high_variable_genes=args.highly_variable_genes
    output_path=args.output_path
    esm_macrogene_amount=args.num_esm_macrogene
    llama_macrogene_amount=args.num_llama_macrogene

    output_folder=args.output_model_path
    task_name=args.task_name
    species_batch_labels=args.species_batch_labels
    input_folder=args.input_model_folder
    encoder_hidden_size=args.encoder_hidden_size
    decoder_hidden_size=args.decoder_hidden_size
    latent_size=args.latent_size
    encoder_dropout_ratio=args.encoder_dropout_ratio
    decoder_dropout_ratio=args.decoder_dropout_ratio
    use_epoch=args.used_epoch

    
    # step1: load the adata and the gene embeddings
    species_to_adata={}
    for i in range(len(species_labels)):
        species_to_adata[species_labels[i]]=sc.read_h5ad(h5ad_files[i])

    species_to_gene_esm_embeddings_path={}
    for i in range(len(species_labels)):
        species_to_gene_esm_embeddings_path[species_labels[i]]=gene_esm_embedding_path[i]

    species_to_gene_llama_embeddings_path={}
    for i in range(len(species_labels)):
        species_to_gene_llama_embeddings_path[species_labels[i]]=gene_llama_embedding_path[i]

    # copy the original data
    species_to_adata_esm = copy.deepcopy(species_to_adata)
    species_to_adata_llama = copy.deepcopy(species_to_adata)

    species_to_gene_esm_embeddings={}
    species_to_gene_llama_embeddings={}
    
    for species, adata in species_to_adata_esm.items():
        adata,species_gene_esm_embeddings=load_gene_embeddings_adata(adata,species=[species],embedding_path=species_to_gene_esm_embeddings_path[species])
        species_to_gene_esm_embeddings.update(species_gene_esm_embeddings)
        species_to_adata_esm[species]=adata
        print('subseting the ', species,' data with the genes with esm embeddings')

    for species, adata in species_to_adata_llama.items():
        adata,species_gene_llama_embeddings=load_gene_embeddings_adata(adata,species=[species],embedding_path=species_to_gene_llama_embeddings_path[species])
        species_to_gene_llama_embeddings.update(species_gene_llama_embeddings)
        species_to_adata_llama[species]=adata
        print('subseting the ', species,' data with the genes with llama embeddings')


    sorted_species_names = sorted(list(species_to_adata.keys()))
    gene_amount_before_hvg_esm = min([v.shape[1] for v in species_to_adata_esm.values()])
    gene_amount_before_hvg_llama = min([v.shape[1] for v in species_to_adata_llama.values()])
    high_variable_genes_esm = min(high_variable_genes,gene_amount_before_hvg_esm)
    high_variable_genes_llama = min(high_variable_genes,gene_amount_before_hvg_llama)

    # step2: subset the adata by the highly variable genes
    species_to_gene_idx_hvg_esm={}
    ct = 0 # this is used to keep track of the index of the gene in the concatenated gene embedding list
    for species in sorted_species_names:
        adata = species_to_adata_esm[species]
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3',n_top_genes=high_variable_genes_esm)
    
        #### here we can set the batch label to find the hvg in each batch later, but for now we are not doing it
        #### could be useful when dealing with nested batches
        hvg_index = adata.var['highly_variable']
        species_to_adata_esm[species] = adata[:, hvg_index]
        species_to_gene_esm_embeddings[species] = species_to_gene_esm_embeddings[species][hvg_index]
        species_to_gene_idx_hvg_esm[species] = (ct,ct+species_to_gene_esm_embeddings[species].shape[0]) # this is the index of the hvg in the concatenated gene list
        ct += species_to_gene_esm_embeddings[species].shape[0]

    species_to_gene_idx_hvg_llama={}
    ct = 0 # this is used to keep track of the index of the gene in the concatenated gene embedding list
    for species in sorted_species_names:
        adata = species_to_adata_llama[species]
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3',n_top_genes=high_variable_genes_llama)
    
        #### here we can set the batch label to find the hvg in each batch later, but for now we are not doing it
        #### could be useful when dealing with nested batches
        hvg_index = adata.var['highly_variable']
        species_to_adata_llama[species] = adata[:, hvg_index]
        species_to_gene_llama_embeddings[species] = species_to_gene_llama_embeddings[species][hvg_index]
        species_to_gene_idx_hvg_llama[species] = (ct,ct+species_to_gene_llama_embeddings[species].shape[0]) # this is the index of the hvg in the concatenated gene list
        ct += species_to_gene_llama_embeddings[species].shape[0]


    # step3: concatenate the gene embeddings
    all_gene_names_esm = []
    for species in sorted_species_names:
        adata = species_to_adata_esm[species]
        species_str = pd.Series([species]*adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        all_gene_names_esm += list(species_str.str.cat(gene_names, sep='_'))
    
    all_gene_names_llama = []
    for species in sorted_species_names:
        adata = species_to_adata_llama[species]
        species_str = pd.Series([species]*adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        all_gene_names_llama += list(species_str.str.cat(gene_names, sep='_'))

    all_gene_esm_embeddings=torch.cat([species_to_gene_esm_embeddings[species] for species in sorted_species_names],dim=0)  # shape is (gene,feature1280)
    all_gene_llama_embeddings=torch.cat([species_to_gene_llama_embeddings[species] for species in sorted_species_names],dim=0)  # shape is (gene,feature4096)

    # step4: initialize the macrogene layer, totally 2000 macrogene
    # the output is {species_gene_name:[weights to 2000 macrogenes]}
    esm_macrogene_weights=macrogene_initialization(all_gene_esm_embeddings,all_gene_names_esm,num_macrogene=esm_macrogene_amount,normalize=False,seed=0)
    llama_macrogene_weights=macrogene_initialization(all_gene_llama_embeddings,all_gene_names_llama,num_macrogene=llama_macrogene_amount,normalize=False,seed=0)
    
    # step5: generate the macrogenes' weight as a list
    esm_centroid_weights = []
    llama_centroid_weights = []
    all_species_gene_names_esm = []
    all_species_gene_names_llama = []
    for species in sorted_species_names:
        adata = species_to_adata_esm[species]
        species_str = pd.Series([species]*adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        species_gene_names = species_str.str.cat(gene_names, sep='_')
        all_species_gene_names_esm = all_species_gene_names_esm+list(species_gene_names)
        for sgn in species_gene_names:
            esm_centroid_weights.append(torch.tensor(esm_macrogene_weights[sgn]))  
    
    for species in sorted_species_names:
        adata = species_to_adata_llama[species]
        species_str = pd.Series([species]*adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        species_gene_names = species_str.str.cat(gene_names, sep='_')
        all_species_gene_names_llama = all_species_gene_names_llama+list(species_gene_names)
        for sgn in species_gene_names:
            llama_centroid_weights.append(torch.tensor(llama_macrogene_weights[sgn]))

    #import pdb;pdb.set_trace()
    # for the macrogene analysis we need to save the esm_macrogene_weights and llama_macrogene_weights
    # with open(metric_dir / f'{run_name}_genes_to_macrogenes.pkl', "wb") as f:
    #        pickle.dump(species_genes_scores_final, f, protocol=4) # Save the scores dict
    import pickle
    with open(output_folder+task_name+'-esm_to_macrogenes.pkl','wb') as f:
        pickle.dump(esm_macrogene_weights,f,protocol=4)

    with open(output_folder+task_name+'-llama_to_macrogenes.pkl','wb') as f:
        pickle.dump(llama_macrogene_weights,f,protocol=4)
    
    #import pdb;pdb.set_trace()
    esm_centroid_weights = torch.stack(esm_centroid_weights)  # shape is gene,2000
    llama_centroid_weights = torch.stack(llama_centroid_weights) # gene, 2000
    
    # step 6 generate the fake macrogene adata
    macrogene_adata={}
    # Calculate the number of species
    num_species = len(sorted_species_names)

    # Calculate the number of high-variable genes per species for LLaMA and ESM
    llama_genes_per_species = llama_centroid_weights.shape[0] // num_species
    esm_genes_per_species = esm_centroid_weights.shape[0] // num_species

    for i in range(num_species):
        species = sorted_species_names[i]
        adata_llama = species_to_adata_llama[species]
        adata_esm = species_to_adata_esm[species]

        # Convert AnnData objects to tensors
        adata_origin_llama = torch.tensor(adata_llama.X.toarray(), dtype=torch.float64)
        adata_origin_esm = torch.tensor(adata_esm.X.toarray(), dtype=torch.float64)

        # Slice the weights for the current species
        llama_start_idx = i * llama_genes_per_species
        llama_end_idx = (i + 1) * llama_genes_per_species
        llama_weights = llama_centroid_weights[llama_start_idx:llama_end_idx, :]

        esm_start_idx = i * esm_genes_per_species
        esm_end_idx = (i + 1) * esm_genes_per_species
        esm_weights = esm_centroid_weights[esm_start_idx:esm_end_idx, :]

        # Compute macrogenes
        llama_macrogene = adata_origin_llama @ llama_weights
        esm_macrogene = adata_origin_esm @ esm_weights

        # Stack the two macrogenes together
        macrogenes_input = torch.cat((esm_macrogene, llama_macrogene), 1)

        # Store the result in a dictionary
        macrogene_adata[species] = ad.AnnData(X=macrogenes_input.numpy(), obs=adata_llama.obs.copy())
        # save the h5ad file
        macrogene_adata[species].write_h5ad(output_folder+task_name+'-'+species+'_macrogene_adata.h5ad')

    #import pdb;pdb.set_trace()
    # for the macrogene analysis, we need to save the macrogene_adata
    #species_label_list=list(macrogene_adata.keys())
    #macrogene_adata[species_label_list[0]].write_h5ad(species_label_list[0]+'_macrogene_adata.h5ad')
    #macrogene_adata[species_label_list[1]].write_h5ad(species_label_list[1]+'_macrogene_adata.h5ad')

    # step6: train model
    species_celltype_labels={species_labels[i]:celltype_labels[i] for i in range(len(species_labels))}
    species_batch_labels=args.batch_labels
    if species_batch_labels =='None':
        species_batch_labels = None
    dataset=SingleCellDataset(species_to_adata=macrogene_adata,species_celltype_labels=species_celltype_labels,species_batch_labels=species_batch_labels)
    #import pdb;pdb.set_trace()

    gene_num = list(dataset.num_genes.values())[0]

    # first get the basal which cource from two species
    finetune_encoder = Encoder(input_size=(esm_macrogene_amount+llama_macrogene_amount),hidden_size=encoder_hidden_size, latent_size=latent_size,dropout_ratio=encoder_dropout_ratio).to(device)
    finetune_decoder = Decoder(gene_num,len(species_labels),hidden_size=decoder_hidden_size,latent_size=latent_size,dropout_ratio=decoder_dropout_ratio).to(device)

    finetune_encoder.load_state_dict(torch.load(input_folder+'esm_llama_encoder-'+str(use_epoch)+'.pth'))
    finetune_decoder.load_state_dict(torch.load(input_folder+'esm_llama_decoder-'+str(use_epoch)+'.pth')) #finetune_aae_supervised_decoder-resilient.pth

    finetune_encoder.eval()
    finetune_decoder.eval()
    
    #import pdb;pdb.set_trace()
    #all_basal=finetune_encoder(torch.tensor(dataset.trans_profiles).to(device))
    batch_size = 5000  # Adjust based on GPU memory
    all_basal = []

    for i in range(0, dataset.trans_profiles.shape[0], batch_size):
        batch = dataset.trans_profiles[i:i+batch_size].to(device)
        with torch.no_grad():  # Disable gradients if not training
            basal_embedding = finetune_encoder(batch)
        all_basal.append(basal_embedding.cpu())  # Move to CPU to free GPU memory
        torch.cuda.empty_cache()

    all_basal = torch.cat(all_basal).to(device)  # Final concatenation on GPU

    all_one_hot_label = torch.tensor(np.vstack(list(dataset.species_onehot_embedding.values())))
    basal_speicies_onehot=torch.cat((all_basal,all_one_hot_label.to(device)),1)
    recon_data=finetune_decoder(basal_speicies_onehot)
    
    # save the basal embedding to a adata object
    adata_basal=AnnData(recon_data.cpu().detach().numpy())
    #list1, list2 = dataset.celltype_labels.values()
    # Combine the two lists into one big list
    #labels = list1.tolist() + list2.tolist()

    labels = [item for sublist in dataset.celltype_labels.values() for item in sublist.tolist()]
    
    id2cell_type = {v:k for k,v in dataset.cell_type_id_dict.items()}
    #import pdb;pdb.set_trace()
    adata_basal.obs['celltype'] = pd.Categorical(labels)
    adata_basal.obs['species'] = [value for sublist in dataset.species_labels.values() for value in sublist]
    adata_basal.obsm['X_esm']=all_basal.cpu().detach().numpy()
    adata_basal.write_h5ad(output_folder+task_name+'-basal.h5ad')

if __name__=="__main__":
    main()
