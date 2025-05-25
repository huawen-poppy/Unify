import torch 
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from dataset import SingleCellDataset
import torch.nn.functional as F

# below i am going to build a macrogene (clusters of the protein embeddings from esm model) supervised aae, which is to add a layer to the aae.
# the layer is to learn the genes to the marocgenes, the weights reflect the protein sequences similarities
# supervised aae is to add a label information to the decoder, the label information is the species information, which is a one-hot vector

class Decoder(nn.Module):
    def __init__(self,input_size,onehot_size,hidden_size=512, latent_size=256,dropout_ratio=0.2):
        super(Decoder,self).__init__()
        self.decoder=nn.Sequential(
            nn.Linear(latent_size+onehot_size,hidden_size), # this is taking the output of the encoder and the one-hot label
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,input_size),
            nn.ReLU()   # add Relu here to make sure the output is non-negative, is to make all the values which is less than 0 is 0, 
            # i add this function based on the fact that our current data have many 0 (min)
        )

    def forward(self,x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self,latent_size=256,hidden_size_adversary=128,species_num=2):
        super(Discriminator,self).__init__()
        self.adversary=nn.Sequential(
            nn.Linear(latent_size,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,species_num),  # this is a classifier, so the output is the number of unique species
        )

    def forward(self,x):
        return self.adversary(x)

class Discriminator_celltype(nn.Module):
    def __init__(self,latent_size=256,hidden_size_adversary=128,celltype_num=12):
        super(Discriminator_celltype,self).__init__()
        self.adversary=nn.Sequential(
            nn.Linear(latent_size,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,hidden_size_adversary),
            nn.BatchNorm1d(hidden_size_adversary),
            nn.ReLU(),
            nn.Linear(hidden_size_adversary,celltype_num),  # this is a classifier, so the output is the number of unique species
            #nn.Linear(latent_size,celltype_num)
        )

    def forward(self,x):
        return self.adversary(x)
    

class Encoder_macrogene(nn.Module):
    def __init__(self, gene_scores,sorted_species_labels_names=None,species_to_gene_idx={},hidden_size=512, latent_size=256,dropout_ratio=0.2):
        super(Encoder_macrogene,self).__init__()
        self.hidden_size=hidden_size
        self.output_size=latent_size
        self.dropout_ratio=dropout_ratio
        self.sorted_species_labels_names=sorted_species_labels_names
        self.num_species=len(sorted_species_labels_names)
        self.num_gene_scores=len(gene_scores)
        self.species_to_gene_idx=species_to_gene_idx
        self.sorted_species_labels_names=sorted(self.species_to_gene_idx.keys())
        self.num_genes=0
        #self.p_weight=nn.Parameter(gene_scores.float().t().log())
        clone_gene_score = gene_scores.clone()
        self.p_weight=nn.Parameter(clone_gene_score.float().t().log())
        self.num_cl=gene_scores.shape[1] # this is the number of the macrogenes (cluster of the protein embeddings from esm model)
        self.cl_layer_norm = nn.LayerNorm(self.num_cl)
        for k,v in self.species_to_gene_idx.items():
            self.num_genes = max(self.num_genes,v[1])
        self.expr_filter = nn.Parameter(torch.zeros(self.num_genes),requires_grad=False)


        self.encoder=nn.Sequential(
            nn.Linear(self.num_cl,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.output_size)
        )
        
        self.p_weights_embedding=nn.Sequential(
            nn.Linear(self.num_cl,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU()
        )

    def forward(self,x,species):
        #import pdb;pdb.set_trace()
        batch_size=x.shape[0]

        # pad the append expr with 0s to fill all gene nodes
        expr = torch.zeros(batch_size,self.num_genes).to(x.device)
        filter_idx = self.species_to_gene_idx[species]
        expr[:,filter_idx[0]:filter_idx[1]] = x
        expr = torch.log(expr + 1)

        # concatenate the gene embeds with the expression as the last item in the embed
        expr = expr.unsqueeze(1)

        # GNN and cluster weights
        clusters = []
        expr_and_genef = expr
        #import pdb;pdb.set_trace()
        encoder_input=nn.functional.linear(expr_and_genef.squeeze(),self.p_weight.exp())
        encoder_input = self.cl_layer_norm(encoder_input)
        encoder_input = F.relu(encoder_input)
        encoder_input = F.dropout(encoder_input,self.dropout_ratio)
        
        encoder_input = encoder_input.squeeze()
        encoded = self.encoder(encoder_input)
        gene_weights_embedding=self.p_weights_embedding(self.p_weight.exp().t())
        return encoded,encoder_input,gene_weights_embedding
    
class finetune_encoder(nn.Module):
    def __init__(self, gene_scores,sorted_species_labels_names=None,species_to_gene_idx={},hidden_size=512, latent_size=256,dropout_ratio=0.2):
        super(finetune_encoder,self).__init__()
        self.hidden_size=hidden_size
        self.output_size=latent_size
        self.dropout_ratio=dropout_ratio
        self.sorted_species_labels_names=sorted_species_labels_names
        self.num_species=len(sorted_species_labels_names)
        self.num_gene_scores=len(gene_scores)
        self.species_to_gene_idx=species_to_gene_idx
        self.sorted_species_labels_names=sorted(self.species_to_gene_idx.keys())
        self.num_genes=0
        self.p_weight=nn.Parameter(gene_scores.float().t().log())
        self.num_cl=gene_scores.shape[1] # this is the number of the macrogenes (cluster of the protein embeddings from esm model)
        self.cl_layer_norm = nn.LayerNorm(self.num_cl)
        for k,v in self.species_to_gene_idx.items():
            self.num_genes = max(self.num_genes,v[1])
        self.expr_filter = nn.Parameter(torch.zeros(self.num_genes),requires_grad=False)


        self.encoder=nn.Sequential(
            nn.Linear(self.num_cl,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Dropout(self.dropout_ratio),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.output_size)
        )
    
    def load_p_weight_from_pretrained_encoder(self, pretrained_p_weight):
        self.p_weight.data = pretrained_p_weight.data


    def forward(self,x,species):
        # x is a list of gene expression data of species [A,B]
        # species is a list of species label [0,1]

        cat_x, tiger_x = x
        cat_label, tiger_label = species

        cat_marco_matrix = torch.matmul(self.p_weight.exp()[:,:cat_x.shape[1]], cat_x)
        tiger_marco_matrix = torch.matmul(self.p_weight.exp()[:,cat_x.shape[1]:], tiger_x)

        joint_marco_matrix = torch.cat([cat_marco_matrix, tiger_marco_matrix], dim=1)
        joint_species_label = torch.cat([cat_label, tiger_label], dim=1)
        # shuffle the joint matrix
        shuffle_array= torch.randperm(joint_marco_matrix.size(0))
        joint_marco_matrix = joint_marco_matrix[shuffle_array]
        joint_species_label = joint_species_label[shuffle_array]

        joint_basal = self.encoder(joint_marco_matrix)

        return joint_basal        
        #import pdb;pdb.set_trace()
    

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, latent_size=256,dropout_ratio=0.2):
        super(Encoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=latent_size
        self.dropout_ratio=dropout_ratio

        self.encoder=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,latent_size)
        )

    def forward(self,x):
        return self.encoder(x)


        