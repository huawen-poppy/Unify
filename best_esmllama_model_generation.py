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
import anndata as ad
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in range(len(grads)):
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in range(len(grads)):
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in range(len(grads)):
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in range(len(grads)):
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def compute_ARI_and_NMI_celltype(y_true, y_feat):
    '''
    y_true: the true cell type labels
    y_feat: the features that we want to cluster
    '''
    n = 20
    resolution = [2 * x / n for x in range(1, n + 1)]
    adata = ad.AnnData(X=y_feat)
    adata.obs['celltype'] = y_true
    best_ari = 0
    for res in resolution:
        sc.tl.pca(adata)
        adata.obsm['X_emb']=adata.X
        sc.pp.neighbors(adata,use_rep='X_emb')
        sc.tl.louvain(adata, resolution=res, key_added='louvain')
        ari = adjusted_rand_score(y_true, adata.obs['louvain'])
        #nmi = normalized_mutual_info_score(y_true, adata.obs['louvain'])
        #sw = silhouette_score(y_feat, adata.obs['louvain'])
        if ari > best_ari:
            best_ari = ari
            #best_nmi = nmi
            #best_sw = sw
        del adata.obs['louvain']
        #print(f'ARI: {ari}, NMI: {nmi}')
    return best_ari#, best_nmi, best_sw

def compute_ARI_and_NMI_celltype_reconstructed(y_true, y_feat):
    '''
    y_true: the true cell type labels
    y_feat: the features that we want to cluster
    '''
    n = 20
    resolution = [2 * x / n for x in range(1, n + 1)]
    adata = ad.AnnData(X=y_feat)
    adata.obs['celltype'] = y_true
    best_ari = 0
    for res in resolution:
        sc.tl.pca(adata)
        #import pdb;pdb.set_trace()
        sc.pp.neighbors(adata,use_rep='X_pca')
        sc.tl.louvain(adata, resolution=res, key_added='louvain')
        ari = adjusted_rand_score(y_true, adata.obs['louvain'])
        #nmi = normalized_mutual_info_score(y_true, adata.obs['louvain'])
        #sw = silhouette_score(y_feat, adata.obs['louvain'])
        if ari > best_ari:
            best_ari = ari
            #best_nmi = nmi
            #best_sw = sw
        del adata.obs['louvain']
        #print(f'ARI: {ari}, NMI: {nmi}')
    return best_ari

def main():
    parser = argparse.ArgumentParser(description='Train the supervised aae with an extra macrogene layer')
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
    parser.add_argument('--evaluate_emb', type=str, help='whether to evaluate the embedding,if false,then evaluate the generation as integration',default="True")

    args = parser.parse_args()
    ## in below code, I am trying to train the supervised aae with an extra macrogene layer
    h5ad_files=args.h5ad_files
    species_labels=args.species_labels
    celltype_labels=args.celltype_labels
    gene_esm_embedding_path=args.gene_esm_embedding_path
    gene_llama_embedding_path=args.gene_llama_embedding_path
    high_variable_genes=args.highly_variable_genes
    output_path=args.output_path
    esm_macrogene_amount=args.num_esm_macrogene
    evaluate_emb=args.evaluate_emb
    llama_macrogene_amount=args.num_llama_macrogene

    config = { 
        'batch_size':256,
        'grad_normalized_type':'l2',
        'hidden_size_adversary_species':256,
        'hidden_size_adversary_celltype':256,
        'latent_size':1024,
        'lr_discriminator_celltype':0.0003823874450101838,
        'lr_discriminator_species':0.00002582115133683638,
        'lr_encoder':0.0006966501901269402,
        'lr_generator':0.00017174110022086215,
        'macrogene_decoder_dropout_ratio':0.1,
        'macrogene_decoder_hidden_size':1024,
        'macrogene_encoder_dropout_ratio':0.2,
        'macrogene_encoder_hidden_size':1024,
        'train_epochs':500
        }
    
    run = wandb.init(project='esmAAE_llama_2k_2k_task8_final',config=config)


    if evaluate_emb == "True":
        evaluate_emb = True
    else:
        evaluate_emb = False
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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


    esm_centroid_weights = torch.stack(esm_centroid_weights)  # shape is gene,2000
    llama_centroid_weights = torch.stack(llama_centroid_weights) # gene, 2000
    # save the preprocess data
    for k,v in species_to_adata_esm.items():
        v.write_h5ad(output_path+'/'+k+'_processed_esm.h5ad')
    for k,v in species_to_adata_llama.items():
        v.write_h5ad(output_path+'/'+k+'_processed_llama.h5ad')

    # step 6 generate the fake macrogene adata
    macrogene_adata={}
    for i in range(len(sorted_species_names)):
        species=sorted_species_names[i]
        adata_llama=species_to_adata_llama[species]
        adata_esm=species_to_adata_esm[species]
        adata_origin_llama=torch.tensor(adata_llama.X,dtype=torch.float64)
        adata_origin_esm=torch.tensor(adata_esm.X,dtype=torch.float64)
        if i == 0:
            llama_weights=llama_centroid_weights[:high_variable_genes_llama,]
            esm_weights=esm_centroid_weights[:high_variable_genes_esm,]
        else:
            llama_weights=llama_centroid_weights[high_variable_genes_llama:,]
            esm_weights=esm_centroid_weights[high_variable_genes_esm:,]
        llama_macrogene=adata_origin_llama @ llama_weights
        esm_macrogene=adata_origin_esm @ esm_weights

        # stack two macrogene together
        macrogenes_input=torch.cat((esm_macrogene,llama_macrogene),1)
        macrogene_adata[species]=ad.AnnData(X=macrogenes_input.numpy(), obs=adata_llama.obs.copy())


    macrogene_encoder_hidden_size=wandb.config['macrogene_encoder_hidden_size']
    macrogene_decoder_hidden_size=wandb.config['macrogene_decoder_hidden_size']
    macrogene_encoder_latent_size=wandb.config['latent_size']
    macrogene_decoder_latent_size=wandb.config['latent_size']
    macrogene_encoder_dropout_ratio=wandb.config['macrogene_encoder_dropout_ratio']
    macrogene_decoder_dropout_ratio=wandb.config['macrogene_decoder_dropout_ratio']
    batch_size=wandb.config['batch_size']
    norm_type=wandb.config['grad_normalized_type']


    #import pdb;pdb.set_trace()
    # step6: train model
    species_celltype_labels={species_labels[i]:celltype_labels[i] for i in range(len(species_labels))}
    species_batch_labels=args.batch_labels
    if species_batch_labels =='None':
        species_batch_labels = None
    dataset=SingleCellDataset(species_to_adata=macrogene_adata,species_celltype_labels=species_celltype_labels,species_batch_labels=species_batch_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    gene_num = list(dataset.num_genes.values())[0]
    print(gene_num)
    import warnings
    warnings.filterwarnings("ignore")

    def evaluation_model_embed(encoder,data=dataset):
        encoder.eval()
        gene_input=data.trans_profiles
        latent=encoder(gene_input.to(device))
        celltype_idx=data.celltype_id_embedding_all
        ari=compute_ARI_and_NMI_celltype(celltype_idx,latent.cpu().detach().numpy())
        return ari

    def evaluation_model_reconstruct(encoder,decoder,data=dataset):
        encoder.eval()
        decoder.eval()
        all_species=sorted(list(data.raw_counts.keys()))
        species_a=all_species[0]
        species_b=all_species[1]
        species_a_origin_gene=data.raw_counts[species_a]
        species_b_origin_gene=data.raw_counts[species_b]
        species_a_cell_num=data.raw_counts[species_a].shape[0]
        species_b_cell_num=data.raw_counts[species_b].shape[0]
        gene_input_for_a=species_a_origin_gene
        gene_input_for_b=species_b_origin_gene
        species_a_onehot=np.unique(data.species_onehot_embedding[species_a])
        species_b_onehot=np.unique(data.species_onehot_embedding[species_b])
        species_a_celltype=data.celltype_id_embedding[species_a]
        species_b_celltype=data.celltype_id_embedding[species_b]
        # speicies a input + species b onehot
        a_latent=encoder(gene_input_for_a.to(device))
        a_latent_speicies_b_onehot=torch.cat((a_latent,torch.tensor([species_b_onehot]*species_a_cell_num).to(device)),1)
        recon_b = decoder(a_latent_speicies_b_onehot)
        #concate the real species b and recon a (fake b) together
        real_b_fake_a=torch.cat((species_b_origin_gene,recon_b.cpu().detach()),0)
        celltype_idx_real_b_fake_a=np.concatenate((species_b_celltype,species_a_celltype),0)
        real_b_fake_a_ari=compute_ARI_and_NMI_celltype_reconstructed(celltype_idx_real_b_fake_a,real_b_fake_a.numpy())

        # speicies b input + species a onehot
        b_latent=encoder(gene_input_for_b.to(device))
        b_latent_speicies_a_onehot=torch.cat((b_latent,torch.tensor([species_a_onehot]*species_b_cell_num).to(device)),1)
        recon_a = decoder(b_latent_speicies_a_onehot)
        #concate the real species a and recon b (fake a) together
        real_a_fake_b=torch.cat((species_a_origin_gene,recon_a.cpu().detach()),0)
        celltype_idx_real_a_fake_b=np.concatenate((species_a_celltype,species_b_celltype),0)
        real_a_fake_b_ari=compute_ARI_and_NMI_celltype_reconstructed(celltype_idx_real_a_fake_b,real_a_fake_b.numpy())
        return real_a_fake_b_ari, real_b_fake_a_ari
    
    hidden_size_adversary_species=wandb.config['hidden_size_adversary_species']
    hidden_size_adversary_celltype=wandb.config['hidden_size_adversary_celltype']
    lr_generator=wandb.config['lr_generator']
    lr_discriminator_species=wandb.config['lr_discriminator_species']
    lr_discriminator_celltype=wandb.config['lr_discriminator_celltype']
    lr_encoder=wandb.config['lr_encoder']
    train_epochs=wandb.config['train_epochs']

    celltype_num=dataset.cell_type_num

    # prepare the model
    encoder = Encoder(input_size=(esm_macrogene_amount+llama_macrogene_amount),hidden_size=macrogene_encoder_hidden_size, latent_size=macrogene_encoder_latent_size,dropout_ratio=macrogene_encoder_dropout_ratio).to(device)
    decoder = Decoder(gene_num,len(species_labels),hidden_size=macrogene_decoder_hidden_size,latent_size=macrogene_decoder_latent_size,dropout_ratio=macrogene_decoder_dropout_ratio).to(device)

    discriminator_species = Discriminator(latent_size=macrogene_encoder_latent_size,hidden_size_adversary=hidden_size_adversary_species,species_num=len(species_labels)).to(device)
    discriminator_celltype = Discriminator_celltype(latent_size=macrogene_encoder_latent_size,hidden_size_adversary=hidden_size_adversary_celltype,celltype_num=celltype_num).to(device)

    # prepare the loss function 
    reconstruction_loss=torch.nn.MSELoss()
    adversarial_species_loss=torch.nn.CrossEntropyLoss()
    adversarial_celltype_loss=torch.nn.CrossEntropyLoss()
    discriminator_species_loss=torch.nn.CrossEntropyLoss()
    discriminator_celltype_loss=torch.nn.CrossEntropyLoss()

    # optimizer 
    import itertools
    optimizer_R=torch.optim.Adam(itertools.chain(encoder.parameters(),decoder.parameters()),lr=lr_generator)
    optimizer_D_species=torch.optim.Adam(discriminator_species.parameters(),lr=lr_discriminator_species)
    optimizer_A=torch.optim.Adam(encoder.parameters(),lr=lr_encoder)
    optimizer_D_celltype=torch.optim.Adam(discriminator_celltype.parameters(),lr=lr_discriminator_celltype)
    # R is for reconstruction loss optimize
    # D_species is for species discriminator optimize
    # A is for the encoder optimize to make it fool the discriminator on the species level
    # C is for the encoder optimize to make it contain the cell type information
    # D_celltype for cell type discriminator optimize

    # put all the model into cuda
    encoder.to(device)
    decoder.to(device)
    discriminator_species.to(device)
    discriminator_celltype.to(device)

    adversarial_species_loss.to(device)
    adversarial_celltype_loss.to(device)
    reconstruction_loss.to(device)
    discriminator_celltype_loss.to(device)
    discriminator_species_loss.to(device)

    print(train_epochs)
    for epoch in range(train_epochs):
        print(epoch)
        Reconstruction_loss = 0
        encoder_loss = 0
        adversary_species_loss = 0
        adversary_celltype_loss = 0
        overall_celltype_precision_score = 0 
        overall_species_precision_score = 0
        overall_celltype_recall_score = 0
        overall_species_recall_score = 0
        overall_celltype_f1_score = 0
        overall_species_f1_score = 0
        overall_celltype_accuracy_score = 0
        overall_species_accuracy_score = 0

        encoder.train()
        decoder.train()
        discriminator_species.train()
        discriminator_celltype.train()

        for batch in dataloader:
            gene, species_onehot, species_idx, celltype_onehot, celltype_idx = batch
            gene = gene.to(device)
            species_idx = species_idx.to(device)
            species_onehot = species_onehot.to(device)
            celltype_onehot = celltype_onehot.to(device)
            celltype_idx = celltype_idx.to(device)
            
            # compute the pareto optimal parameters for the entire loss function
            optimizer_R.zero_grad()
            encoder.train()
            discriminator_species.eval()
            discriminator_celltype.eval()

            basal = encoder(gene)
            basal_species_onehot=torch.cat((basal,species_onehot),1)
            recon = decoder(basal_species_onehot)
                
            # reconstruction loss
            recon_loss = reconstruction_loss(gene, recon)
            recon_loss_data = recon_loss.item()
                
            # compute the grad
            grad_encoder_recon=[]
            recon_loss.backward()
            for param in encoder.parameters():
                grad_encoder_recon.append(param.grad.data.clone())
                
            # after get the grad from encoder, clean all pre computed grad for encoder and decoder
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator_celltype.zero_grad()
            discriminator_species.zero_grad()

            # get the grad encoder for the discriminator species
            optimizer_A.zero_grad()
            encoder.train()
            discriminator_species.eval()
            discriminator_celltype.eval()
            fake_latent = encoder(gene)
            validity_fake_latent = discriminator_species(fake_latent)
            validity_fake_latent_celltype = discriminator_celltype(fake_latent)
            adv_loss = adversarial_species_loss(validity_fake_latent, species_idx)
            adv_loss_new=-adv_loss
            adv_species_loss_data = -1*adv_loss.item()
            adv_loss_new.backward()
            grad_encoder_adv_species=[]
            for param in encoder.parameters():
                if param.grad is not None:
                    # we update the grad to the negative of the grad to better align the MGDA algorithm 
                    grad_encoder_adv_species.append(param.grad.data.clone())
            
            # clean all the grad for encoder and decoder
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator_celltype.zero_grad()
            discriminator_species.zero_grad()

            # get the grad encoder for the discriminator celltype
            optimizer_A.zero_grad()
            encoder.train()
            discriminator_species.eval()
            discriminator_celltype.eval()
            fake_latent = encoder(gene)
            validity_fake_latent_celltype = discriminator_celltype(fake_latent)
            celltype_loss = adversarial_celltype_loss(validity_fake_latent_celltype, celltype_idx)
            adv_celltype_loss_data = celltype_loss.item()
            celltype_loss.backward()

            grad_encoder_adv_celltype=[]
            for param in encoder.parameters():
                if param.grad is not None:
                    grad_encoder_adv_celltype.append(param.grad.data.clone())
            # clean all the grad 
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator_celltype.zero_grad()
            discriminator_species.zero_grad()

            # normalize all gradients
            gn =gradient_normalizers([grad_encoder_recon,grad_encoder_adv_species,grad_encoder_adv_celltype], [recon_loss_data,adv_species_loss_data,adv_celltype_loss_data], norm_type)
            grads=[grad_encoder_recon,grad_encoder_adv_species,grad_encoder_adv_celltype]
            for t in range(3):
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i]/gn[t]
                
            sol, min_norm = MinNormSolver.find_min_norm_element(grads)
            
            alpha, beta, gamma = float(sol[0]), float(sol[1]), float(sol[2])
            # Renorm the sol to ensure that celltype loss are the dominant loss
            gamma = max(0.4, gamma)
            alpha_prop = alpha/(alpha+beta)
            beta_prop = beta/(alpha+beta)
            new_sum = 1 - gamma
            alpha = alpha_prop*new_sum
            beta = beta_prop*new_sum

            # Optimize the reconstruction loss
            optimizer_R.zero_grad()
            encoder.train()
            discriminator_species.eval()
            discriminator_celltype.eval()
            fake_latent = encoder(gene)
            fake_latent_speicies_onehot=torch.cat((fake_latent,species_onehot),1)
            recon = decoder(fake_latent_speicies_onehot)

            recon_loss = reconstruction_loss(gene, recon)
            # note that the hyperparameter can be changed
            R_loss = alpha*recon_loss
            Reconstruction_loss += R_loss.item()
            #wandb.log({"Reconstruction_loss": recon_loss.item()})
            #print(f'Reconstruction_loss: {R_loss.item()}')
            R_loss.backward()
            optimizer_R.step()

            # Optimize the basal state (for encoder only)
            optimizer_A.zero_grad()
            encoder.train()
            discriminator_species.eval()
            discriminator_celltype.eval()
            fake_latent = encoder(gene)
            #import pdb; pdb.set_trace()
            validity_fake_latent = discriminator_species(fake_latent)
            validity_fake_latent_celltype = discriminator_celltype(fake_latent)
            #import pdb; pdb.set_trace()
            adv_loss = adversarial_species_loss(validity_fake_latent, species_idx)
            celltype_loss = adversarial_celltype_loss(validity_fake_latent_celltype, celltype_idx)
            A_loss = beta*adv_loss*(-1) + gamma*celltype_loss
            encoder_loss += A_loss.item()
            A_loss.backward()
            optimizer_A.step()
            #wandb.log({"species_adv_loss": adv_loss.item()})
            #print(f'species_adv_loss: {adv_loss.item()}')       

            # for the species discriminator
            encoder.eval()
            discriminator_celltype.eval()
            discriminator_species.train()
            optimizer_D_species.zero_grad()
            fake_latent = encoder(gene)
            validity_fake_latent = discriminator_species(fake_latent)
            predicted_species_labels = np.argmax(validity_fake_latent.cpu().detach().numpy(), axis=1)
            species_precision, species_recall, species_f1, species_accuracy=compute_metrics(species_idx.cpu().detach().numpy(), predicted_species_labels)
            overall_species_accuracy_score += species_accuracy
            overall_species_f1_score += species_f1
            overall_species_precision_score += species_precision
            overall_species_recall_score += species_recall

            #import pdb; pdb.set_trace()
            D_loss = discriminator_species_loss(validity_fake_latent, species_idx)
            D_loss.backward()
            adversary_species_loss += D_loss.item()
            optimizer_D_species.step()
            #wandb.log({"discriminator_species_loss": D_loss.item()})
            #print(f'discriminator_species_loss: {D_loss.item()}')

            # for the celltype discriminator
            encoder.eval()
            discriminator_species.eval()
            discriminator_celltype.train()

            optimizer_D_celltype.zero_grad()
            fake_latent = encoder(gene)
            validity_fake_latent = discriminator_celltype(fake_latent)
            predicted_celltype_labels = np.argmax(validity_fake_latent.cpu().detach().numpy(), axis=1)
            celltype_precision, celltype_recall, celltype_f1, celltype_accuracy=compute_metrics(celltype_idx.cpu().detach().numpy(), predicted_celltype_labels)
            overall_celltype_accuracy_score += celltype_accuracy
            overall_celltype_f1_score += celltype_f1
            overall_celltype_precision_score += celltype_precision
            overall_celltype_recall_score += celltype_recall

            D_celltype_loss = discriminator_celltype_loss(validity_fake_latent, celltype_idx)
            D_celltype_loss.backward()
            adversary_celltype_loss += D_celltype_loss.item()
            optimizer_D_celltype.step()
            #wandb.log({"discriminator_celltype_loss": D_celltype_loss.item()})
            #print(f'discriminator_celltype_loss: {D_celltype_loss.item()}')
            #predicted_labels = np.argmax(softmax_output, axis=1)

                
        # compute the metric for the entire epoch
        if epoch == 0:
            overall_celltype_ARI=0
            real_a_overall_celltype_ARI=0
            real_b_overall_celltype_ARI=0 
        if (epoch+1)%5==0:
            overall_celltype_ARI_new = evaluation_model_embed(encoder,data=dataset)
            #real_a_overall_celltype_ARI, real_b_overall_celltype_ARI = evaluation_model_reconstruct(encoder,decoder,data=dataset)
            print(f'overall_celltype_ari_EMB: {overall_celltype_ARI}')
            # print(f'overall_a_celltype_ARI: {real_a_overall_celltype_ARI}')
            # print(f'overall_b_celltype_ARI: {real_b_overall_celltype_ARI}')
            encoder.eval()
            decoder.eval()
            discriminator_celltype.eval()
            discriminator_species.eval()
            if overall_celltype_ARI_new>overall_celltype_ARI:
                torch.save(encoder.state_dict(), output_path+'/esm_llama_encoder-'+str(epoch)+'.pth')
                torch.save(decoder.state_dict(), output_path+'/esm_llama_decoder-'+str(epoch)+'.pth')
                torch.save(discriminator_species.state_dict(),output_path+'/esm_llama_discriminator_species-'+str(epoch)+'.pth')
                torch.save(discriminator_celltype.state_dict(), output_path+'/esm_llama_discriminator_celltype-'+str(epoch)+'.pth')
            overall_celltype_ARI = overall_celltype_ARI_new

        Autoencoder_generator_loss=Reconstruction_loss/len(dataloader)
        Encoder_loss=encoder_loss/len(dataloader)
        Adversary_celltype_loss=adversary_celltype_loss/len(dataloader)
        Adversary_species_loss=adversary_species_loss/len(dataloader)
        Target_loss=Autoencoder_generator_loss+Encoder_loss
        overall_celltype_precision_score=overall_celltype_precision_score/len(dataloader)
        overall_species_precision_score=overall_species_precision_score/len(dataloader)
        overall_celltype_recall_score=overall_celltype_recall_score/len(dataloader)
        overall_species_recall_score=overall_species_recall_score/len(dataloader)
        overall_celltype_f1_score=overall_celltype_f1_score/len(dataloader)
        overall_species_f1_score=overall_species_f1_score/len(dataloader)
        overall_celltype_accuracy_score=overall_celltype_accuracy_score/len(dataloader)
        overall_species_accuracy_score=overall_species_accuracy_score/len(dataloader)
        
        wandb.log(
            {
                'Reconstruction_loss': Autoencoder_generator_loss, 
                'overall distriminators loss': Encoder_loss,
                'Adversary_species_loss': Adversary_species_loss,
                'Adversary_celltype_loss': Adversary_celltype_loss,
                'autoencoder+encoder (target) loss': Target_loss,
                'overall_celltype_precision_score': overall_celltype_precision_score,
                'overall_species_precision_score': overall_species_precision_score,
                'overall_celltype_recall_score': overall_celltype_recall_score,
                'overall_species_recall_score': overall_species_recall_score,
                'overall_celltype_f1_score': overall_celltype_f1_score,
                'overall_species_f1_score': overall_species_f1_score,
                'overall_celltype_accuracy_score': overall_celltype_accuracy_score,
                'overall_species_accuracy_score': overall_species_accuracy_score,
                'overall_celltype_ARI_on_embeddings': overall_celltype_ARI
                }
            )
           
        print(f'Epoch {epoch}, AE Loss: {Reconstruction_loss/len(dataloader)}\nAdversary species Loss: {adversary_species_loss/len(dataloader)}\nAdversary celltype Loss: {adversary_celltype_loss/len(dataloader)}\n')

main()
