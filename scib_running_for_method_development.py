# use the scib-pipeline-R4.0 environment
import os
import sys
import copy
from anndata import AnnData
import glob
import anndata as ad
import scib
import argparse
import pandas as pd
import numpy as np
import re
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def convert_to_days(value):
    if 'Day' in value:
        return int(value.replace('Day', ''))
    elif 'Week' in value:
        return int(value.replace('Week', '')) * 7
    elif 'Month' in value:
        return int(value.replace('Month', '')) * 30
    elif 'hpf' in value:
        return int(value.replace('hpf', '')) / 24
    else:
        return None

def compute_ARI_and_NMI_species(adata,batch_label='species'):
    n = 20
    resolution = [2 * x / n for x in range(1, n + 1)]
    y_true =adata.obs[batch_label] 
    best_ari = 0
    best_nmi = 0
    for res in resolution:
        sc.tl.pca(adata)
        sc.tl.louvain(adata, resolution=res, key_added='louvain')
        ari = adjusted_rand_score(y_true, adata.obs['louvain'])
        nmi = normalized_mutual_info_score(y_true, adata.obs['louvain'])
        if ari > best_ari:
            best_ari = ari
            best_nmi = nmi
        del adata.obs['louvain']
        #print(f'ARI: {ari}, NMI: {nmi}')
    return best_ari, best_nmi

import warnings
warnings.filterwarnings("ignore")
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='scib')
    parser.add_argument('--input_folder',type=str,help='the input folder name for the model output')
    parser.add_argument('--processed_data_folder',type=str,help='the input folder name for the processed data')
    parser.add_argument('--target',type=str,help='the task id')
    parser.add_argument('--file_name',type=str,help='the file name',default=None)
    parser.add_argument('--save_dir',type=str,help='the save folder name')

    args = parser.parse_args()
    input_folder=args.input_folder
    processed_data_folder=args.processed_data_folder
    target=args.target
    pre_dir=args.input_folder
    file_name=args.file_name

    if input_folder.endswith('/')==False:
        input_folder=input_folder+'/'
    if pre_dir.endswith('/')==False:
        save_dir=pre_dir+'/'
    else:
        save_dir=pre_dir

    save_dir_output=args.save_dir

    if file_name is not None:   
        files = [file_name] 
    else:
        files = [f for f in os.listdir(input_folder) if f.startswith(target)]

    for file in files:
        print(file)
        #import pdb;pdb.set_trace()
        adata=sc.read_h5ad(input_folder+file)
        adata.X=np.nan_to_num(adata.X, nan=0.0)
        #sc.pp.normalize_total(adata)
        #sc.pp.log1p(adata)

        if 'basal.h5ad' in file:
            emb_label='X_esm'
            hvg_label=False
            type_label='embed'
            order_needed=False
            full_data_contained=False
            species_a_name=None
            one2one_usage=True
        else:
            one2one_usage=False
            order_needed=True
            emb_label='X_pca'
            hvg_label=False
            type_label='full'
            full_data_contained=False
            pattern = r'real_(\w+)_fake_(\w+)'
            match = re.findall(pattern, file)
            species_a_name=match[0][0]

        all_raw_files=glob.glob(processed_data_folder+target+'_*.h5ad',recursive=True)
        #all_raw_files=glob.glob('/ibex/scratch/projects/c2101/species_integration/data/task8'+'*.h5ad',recursive=True)
        all_raw_files=sorted(all_raw_files)
        print(all_raw_files[0])
        print(all_raw_files[1])
        if species_a_name is None:
            adata_a=sc.read_h5ad(all_raw_files[0])
            adata_b=sc.read_h5ad(all_raw_files[1])
        elif species_a_name in all_raw_files[0]:
            adata_a=sc.read_h5ad(all_raw_files[0])
            adata_b=sc.read_h5ad(all_raw_files[1])
        else:
            adata_a=sc.read_h5ad(all_raw_files[1])
            adata_b=sc.read_h5ad(all_raw_files[0])
        
        sc.pp.normalize_total(adata_a)
        sc.pp.log1p(adata_a)
        sc.pp.highly_variable_genes(adata_a)
        sc.tl.pca(adata_a)
        sc.pp.neighbors(adata_a)
        adata_a.uns["iroot"] = 0
        sc.tl.diffmap(adata_a)
        sc.tl.dpt(adata_a)

        sc.pp.normalize_total(adata_b)
        sc.pp.log1p(adata_b)
        sc.pp.highly_variable_genes(adata_b)
        sc.tl.pca(adata_b)
        sc.pp.neighbors(adata_b)
        adata_b.uns["iroot"] = 0
        sc.tl.diffmap(adata_b)
        sc.tl.dpt(adata_b)
         
        if target in ['task6']:
            mapping=pd.read_csv('/ibex/scratch/projects/c2101/benchmark/one2one_orthologs/nema_sty.csv')
            mapping.drop_duplicates(inplace=True,ignore_index=True)
            mapping.drop_duplicates(subset=['gene_name'],inplace=True,ignore_index=True)
            mapping.drop_duplicates(subset=['Stylophora'],inplace=True,ignore_index=True)
            if species_a_name == 'sty':
                nema_overlap=list(set(adata_b.var_names) & set(mapping['gene_name']))
                mapping=mapping.loc[mapping['gene_name'].isin(nema_overlap)]  
                target_sty_names=adata_a.var_names.str.replace('.{2}$','')
                adata_a.var_names=target_sty_names   
                mapping=mapping.loc[mapping['Stylophora'].isin(adata_a.var_names)]  
                gene_mapping = dict(zip(mapping['gene_name'], mapping['Stylophora']))
                adata_b.var.index = adata_b.var.index.map(lambda x: gene_mapping.get(x, x))
            elif species_a_name == 'nema':
                nema_overlap=list(set(adata_a.var_names) & set(mapping['gene_name']))
                mapping=mapping.loc[mapping['gene_name'].isin(nema_overlap)]  
                target_sty_names=adata_b.var_names.str.replace('.{2}$','')
                adata_b.var_names=target_sty_names   
                mapping=mapping.loc[mapping['Stylophora'].isin(adata_b.var_names)]  
                gene_mapping = dict(zip(mapping['Stylophora'], mapping['gene_name']))
                adata_b.var.index = adata_b.var.index.map(lambda x: gene_mapping.get(x, x))
            else:
                nema_overlap=list(set(adata_a.var_names) & set(mapping['gene_name']))
                adata_a=adata_a[:,adata_a.var_names.isin(mapping['gene_name'])]  
                mapping=mapping.loc[mapping['gene_name'].isin(nema_overlap)]  
                target_sty_names=adata_b.var_names.str.replace('.{2}$','')
                adata_b.var_names=target_sty_names   
                adata_b=adata_b[:,adata_b.var_names.isin(mapping['Stylophora'])]  
                mapping=mapping.loc[mapping['Stylophora'].isin(adata_b.var_names)]  
                mapping=mapping.set_index('Stylophora')
                mapping=mapping.loc[adata_b.var_names]
                adata_b.var_names=mapping['gene_name']
                common_genes=list(set(adata_a.var_names) & set(adata_b.var_names))   
                adata_a=adata_a[:,adata_a.var_names.isin(common_genes)]
                adata_b=adata_b[:,adata_b.var_names.isin(common_genes)]

        elif target in ['task8','task8-1','task8-2']:
            mapping=pd.read_csv('/ibex/scratch/projects/c2101/benchmark/one2one_orthologs/fish_to_frog.csv')
            mapping.drop_duplicates(inplace=True,ignore_index=True)
            mapping.drop_duplicates(subset=['fish_gene'],inplace=True,ignore_index=True)
            mapping.drop_duplicates(subset=['frog_gene'],inplace=True,ignore_index=True)
            if species_a_name == 'frog':
                frog_overlap=list(set(adata_a.var_names) & set(mapping['frog_gene']))
                mapping=mapping.loc[mapping['frog_gene'].isin(frog_overlap)]  
                mapping=mapping.loc[mapping['fish_gene'].isin(adata_b.var_names)]  
                gene_mapping = dict(zip(mapping['fish_gene'], mapping['frog_gene']))
                adata_b.var.index = adata_b.var.index.map(lambda x: gene_mapping.get(x, x))
            elif species_a_name == 'fish':
                fish_overlap=list(set(adata_a.var_names) & set(mapping['fish_gene']))
                mapping=mapping.loc[mapping['fish_gene'].isin(fish_overlap)]  
                mapping=mapping.loc[mapping['frog_gene'].isin(adata_b.var_names)]  
                gene_mapping = dict(zip(mapping['frog_gene'], mapping['fish_gene']))
                adata_b.var.index = adata_b.var.index.map(lambda x: gene_mapping.get(x, x))
            else:
                frog_overlap=list(set(adata_b.var_names) & set(mapping['frog_gene']))
                adata_b=adata_b[:,adata_b.var_names.isin(mapping['frog_gene'])]  # get 67 genes
                mapping=mapping.loc[mapping['frog_gene'].isin(frog_overlap)]
                adata_a=adata_a[:,adata_a.var_names.isin(mapping['fish_gene'])]
                mapping=mapping.set_index('frog_gene')
                mapping=mapping.loc[adata_b.var_names]
                adata_b.var_names=mapping['fish_gene']
                common_genes=list(set(adata_b.var_names) & set(adata_a.var_names))   
                adata_a=adata_a[:,adata_a.var_names.isin(common_genes)]
                adata_b=adata_b[:,adata_b.var_names.isin(common_genes)]

        elif target in ['task10']:
            mapping=pd.read_csv('/ibex/scratch/projects/c2101/benchmark/one2one_orthologs/fish_to_fly.csv')
            mapping.drop_duplicates(inplace=True,ignore_index=True)    
            if species_a_name == 'fish':
                fish_overlap=list(set(adata_a.var_names) & set(mapping['fish_gene']))
                mapping=mapping.loc[mapping['fish_gene'].isin(fish_overlap)]  
                mapping=mapping.loc[mapping['fly_symbol'].isin(adata_b.var_names)]  
                gene_mapping = dict(zip(mapping['fly_symbol'], mapping['fish_gene']))
                adata_b.var.index = adata_b.var.index.map(lambda x: gene_mapping.get(x, x))

            elif species_a_name == 'fly':
                fish_overlap=list(set(adata_b.var_names) & set(mapping['fish_gene']))
                mapping=mapping.loc[mapping['fish_gene'].isin(fish_overlap)]  
                mapping=mapping.loc[mapping['fly_symbol'].isin(adata_a.var_names)]  
                gene_mapping = dict(zip(mapping['fish_gene'], mapping['fly_symbol']))
                adata_b.var.index = adata_b.var.index.map(lambda x: gene_mapping.get(x, x))
            else:
                ################## convert the fish gene to fly orthologous #############
                fish_overlap=list(set(adata_a.var_names) & set(mapping['fish_gene']))
                adata_a=adata_a[:,adata_a.var_names.isin(mapping['fish_gene'])]  # get 296 genes
                mapping=mapping.loc[mapping['fish_gene'].isin(fish_overlap)]  # 296 gene pairs
                fly_overlap=list(set(adata_b.var_names) & set(mapping['fly_symbol']))
                adata_b=adata_b[:,adata_b.var_names.isin(mapping['fly_symbol'])]  # get 296 genes
                mapping=mapping.loc[mapping['fly_symbol'].isin(fly_overlap)] 

                mapping=mapping.set_index('fly_symbol')
                mapping=mapping.loc[adata_b.var_names]
                target_orth_fish_names=mapping['fish_gene']
                adata_b.var_names=target_orth_fish_names   # convert both to the fish gene symbol
                
                common_genes=list(set(adata_b.var_names) & set(adata_a.var_names))   
                adata_a=adata_a[:,adata_a.var_names.isin(common_genes)]
                adata_b=adata_b[:,adata_b.var_names.isin(common_genes)]
        

        elif target in ['task9','task9-1']:
            mapping=pd.read_csv('/ibex/scratch/projects/c2101/benchmark/one2one_orthologs/ant_to_mouse.csv')
            mapping=mapping.iloc[:,3:5]
            mapping.drop_duplicates(inplace=True,ignore_index=True)

            # remove all the duplicated genes in mouse and human in the mapping file    
            mapping.drop_duplicates(subset=['mouse_gene'],inplace=True,ignore_index=True)
            mapping.drop_duplicates(subset=['ant_gene'],inplace=True,ignore_index=True)
            temp_name=mapping['ant_gene'].str.replace('_','-')
            mapping['ant_gene']=temp_name

            ################## convert the ant gene to mouse orthologous #############
            ant_overlap=list(set(adata_a.var_names) & set(mapping['ant_gene']))
            adata_a=adata_a[:,adata_a.var_names.isin(ant_overlap)]# get 103 genes
            mapping=mapping.loc[mapping['ant_gene'].isin(ant_overlap)]  # 103 gene pairs
            mapping=mapping.set_index('ant_gene')
            mapping=mapping.loc[adata_a.var_names]

            adata_a.var_names=mapping['mouse_gene']
            common_genes=list(set(adata_b.var_names) & set(adata_a.var_names))
            adata_a=adata_a[:,adata_a.var_names.isin(common_genes)]
            adata_b=adata_b[:,adata_b.var_names.isin(common_genes)]

        elif target in ['task12']:
            mapping=pd.read_csv('/ibex/scratch/projects/c2101/benchmark/one2one_orthologs/ciona_to_nema.csv')
            mapping.drop_duplicates(inplace=True,ignore_index=True)    # get 315 pairs
            ciona_overlap=list(set(adata_a.var_names) & set(mapping['ciona_gene']))
            adata_a=adata_a[:,adata_a.var_names.isin(mapping['ciona_gene'])] # get 296 genes
            mapping=mapping.loc[mapping['ciona_gene'].isin(ciona_overlap)]   # 296 gene pairs

            nema_overlap=list(set(adata_b.var_names) & set(mapping['nema_gene']))
            adata_b=adata_b[:,adata_b.var_names.isin(mapping['nema_gene'])] # get 296 genes
            mapping=mapping.loc[mapping['nema_gene'].isin(nema_overlap)]
            mapping=mapping.set_index('nema_gene')
            mapping=mapping.loc[adata_b.var_names]
            target_orth_names=mapping['ciona_gene']

            adata_b.var_names=target_orth_names   # convert both to the fish gene symbol
            adata_b.var_names_make_unique()
            common_genes=list(set(adata_b.var_names) & set(adata_a.var_names))
            adata_a=adata_a[:,adata_a.var_names.isin(common_genes)]
            adata_b=adata_b[:,adata_b.var_names.isin(common_genes)]

        elif target in ['task27']:
            mapping=pd.read_csv('/ibex/scratch/projects/c2101/benchmark/one2one_orthologs/pig_to_MM.csv')
            mapping.drop_duplicates(inplace=True,ignore_index=True)
            a_name='MM_gene'
            b_name='pig_gene'
            mapping = mapping.drop_duplicates(subset=a_name)
            a_overlap=list(set(adata_a.var_names) & set(mapping[a_name]))
            adata_a=adata_a[:,adata_a.var_names.isin(mapping[a_name])]
            mapping=mapping.loc[mapping[a_name].isin(a_overlap)]
            b_overlap=list(set(adata_b.var_names) & set(mapping[b_name]))
            adata_b=adata_b[:,adata_b.var_names.isin(mapping[b_name])]
            mapping=mapping.loc[mapping[b_name].isin(b_overlap)]
            adata_a=adata_a[:,adata_a.var_names.isin(mapping[a_name])]
            mapping = mapping.drop_duplicates(subset=b_name)
            mapping=mapping.set_index(b_name)
            mapping=mapping.loc[adata_b.var_names]
            target_orth_names=mapping[a_name]
            adata_b.var_names=target_orth_names   # convert both to the fish gene symbol
            common_genes=list(set(adata_b.var_names) & set(adata_a.var_names))
            adata_a=adata_a[:,adata_a.var_names.isin(common_genes)]
            adata_b=adata_b[:,adata_b.var_names.isin(common_genes)]


        if target in ['task81','task81-1','task81-2']:
            adata_a.obs['True_time']=adata_a.obs['TimeID'].str.replace('hpf','').astype('float32')
            adata_b.obs['True_time']=adata_b.obs['stage'].str.replace('Stage_','').astype('float32')
            adata_a.obs['True_time'] = (adata_a.obs['True_time'] - adata_a.obs['True_time'].min()) / (adata_a.obs['True_time'].max() - adata_a.obs['True_time'].min())
            adata_b.obs['True_time'] = (adata_b.obs['True_time'] - adata_b.obs['True_time'].min()) / (adata_b.obs['True_time'].max() - adata_b.obs['True_time'].min())
            
        if target == 'task10':
            adata_a.obs['True_time'] = adata_a.obs['stage'].apply(convert_to_days).astype('float32')
            adata_b.obs['True_time'] = adata_b.obs['stage'].str.extract(r'D(\d+).*', expand=False).astype('float32')
            adata_a.obs['True_time'] = (adata_a.obs['True_time'] - adata_a.obs['True_time'].min()) / (adata_a.obs['True_time'].max() - adata_a.obs['True_time'].min())
            adata_b.obs['True_time'] = (adata_b.obs['True_time'] - adata_b.obs['True_time'].min()) / (adata_b.obs['True_time'].max() - adata_b.obs['True_time'].min())

        if target in ['task7','task3']:
            celltype_label='NewCelltype'
        elif target in ['task8','task8-1','task8-2','task27',]:
            celltype_label='cell_type'
        elif target in ['task10','task12']:
            celltype_label='cell_lineage'
        elif target in ['task6','task9','task9-1']:
            celltype_label='celltype'
        

        if one2one_usage:
            adata_ls = ad.concat([adata_a,adata_b],join='inner')
            adata.obs_names=adata_ls.obs_names
        elif order_needed:
            shared_genes=list(set(adata_a.var_names) & set(adata_b.var_names))
            # create a adata object with the same shape of adata_b
            adata_b_new_X=np.zeros(adata_b.shape)
            # track the gene index of the shared genes in adata_a 
            shared_genes_index_a=[]
            shared_genes_index_b=[]
            for gene in shared_genes:
                if gene in adata_a.var_names and gene in adata_b.var_names:
                    idx_a=adata_a.var_names.get_loc(gene)
                    idx_b=adata_b.var_names.get_loc(gene)
                    if isinstance(idx_a,int) and isinstance(idx_b,int):
                        shared_genes_index_a.append(idx_a)
                        shared_genes_index_b.append(idx_b)
            #shared_genes_index_a=[adata_a.var_names.get_loc(gene) for gene in shared_genes]
            # track the gene index of the shared genes in adata_b
            #shared_genes_index_b=[adata_b.var_names.get_loc(gene) for gene in shared_genes]
            adata_b_new_X[:,shared_genes_index_a]=adata_b.X[:,shared_genes_index_b]
            adata_b_new=AnnData(adata_b_new_X)
            adata_b_new.var_names=adata_a.var_names
            adata_b_new.obs_names=adata_b.obs_names
            adata_b_new.obs=adata_b.obs
            adata_b=adata_b_new
            adata_ls = ad.concat([adata_a,adata_b],join='inner')
            adata.obs_names=adata_ls.obs_names
            #import pdb;pdb.set_trace()
        adata_ls.X=np.nan_to_num(adata_ls.X, nan=0.0)
        adata_ls.obs_names_make_unique()
        adata_ls.obs['batch']=adata_ls.obs['orig.ident']  # for the new dataset, use batch, old dataset use orig.ident
        adata_ls.obs['celltype']=adata_ls.obs[celltype_label]
        adata.obs['batch']=adata.obs['species']

        if emb_label not in adata.obsm.keys():
            sc.tl.pca(adata,svd_solver="arpack") 
        sc.pp.neighbors(adata,use_rep=emb_label)
        
        adata.obs_names_make_unique()
        adata_ls.obs_names_make_unique()
        adata.obs['batch']=adata.obs['batch'].astype('category')
        adata.obs['celltype']=adata.obs['celltype'].astype('category')
        adata_ls.obs['batch']=adata_ls.obs['batch'].astype('category')
        adata_ls.obs['celltype']=adata_ls.obs['celltype'].astype('category')

        if emb_label == 'X_pca':
            adata.obs_names=adata_ls.obs_names.to_list()
            adata.var_names=adata_ls.var_names.to_list()
        else:
            adata.obsm['X_emb']=adata.obsm[emb_label]
        
        df=scib.metrics.metrics(adata_ls, adata, batch_key='batch', label_key='celltype', embed=emb_label, cluster_key=None, cluster_nmi=None,ari_=True, nmi_=True, nmi_method='arithmetic', nmi_dir=None, silhouette_=True, si_metric='euclidean',pcr_=True, cell_cycle_=False, organism='mouse', hvg_score_=hvg_label, isolated_labels_=True,isolated_labels_f1_=True, isolated_labels_asw_=True, n_isolated=True, graph_conn_=True,trajectory_=True, kBET_=True, lisi_graph_=True, ilisi_=True, clisi_=True, subsample=0.5, n_cores=30,type_=type_label,verbose=False)
        ari_batch,nmi_batch=compute_ARI_and_NMI_species(adata,batch_label='batch')
        df.loc['ARI_batch']=1-ari_batch
        df.loc['NMI_batch']=1-nmi_batch
        
        df.to_csv(save_dir_output+file.replace('.h5ad','_scib_output_new.csv'))
        print('metric finish!')
        
        if full_data_contained:
            adata=sc.read_h5ad(input_folder+file)
            adata_a=sc.read_h5ad(all_raw_files[0])
            adata_b=sc.read_h5ad(all_raw_files[1])
            adata_a.var_names=np.unique(adata.obs['species'])[0]+'_'+adata_a.var_names
            adata_b.var_names=np.unique(adata.obs['species'])[1]+'_'+adata_b.var_names
            sc.pp.normalize_total(adata_a)
            sc.pp.log1p(adata_a)
            sc.pp.highly_variable_genes(adata_a)
            sc.tl.pca(adata_a)
            sc.pp.neighbors(adata_a)
            adata_a.uns["iroot"] = 0
            sc.tl.diffmap(adata_a)
            sc.tl.dpt(adata_a)

            sc.pp.normalize_total(adata_b)
            sc.pp.log1p(adata_b)
            sc.pp.highly_variable_genes(adata_b)
            sc.tl.pca(adata_b)
            sc.pp.neighbors(adata_b)
            adata_b.uns["iroot"] = 0
            sc.tl.diffmap(adata_b)
            sc.tl.dpt(adata_b)

            recon_a=adata[:np.unique(adata.obs['species'],return_counts=True)[1][0],]
            recon_b=adata[np.unique(adata.obs['species'],return_counts=True)[1][0]:,]
            recon_a.obs_names=adata_a.obs_names
            recon_b.obs_names=adata_b.obs_names
            recon_a.var_names=adata_a.var_names
            recon_b.var_names=adata_b.var_names
            sc.pp.normalize_total(recon_a)
            sc.pp.log1p(recon_a)
            sc.pp.normalize_total(recon_b)
            sc.pp.log1p(recon_b)
            adata=ad.concat([recon_a,recon_b],join='outer')
            adata.X=np.nan_to_num(adata.X, nan=0.0)
            
            emb_label='X_pca'
            hvg_label=True
            type_label='full'
            adata_ls = ad.concat([adata_a,adata_b],join='outer')
            adata_ls.X=np.nan_to_num(adata_ls.X, nan=0.0)
            adata_ls.obs_names_make_unique()
            adata.obs_names_make_unique()
            adata_ls.obs['batch']=adata_ls.obs['orig.ident']  # for the new dataset, use batch, old dataset use orig.ident
            adata_ls.obs['celltype']=adata_ls.obs[celltype_label]
            adata.obs['batch']=adata.obs['species']
            sc.tl.pca(adata,svd_solver="arpack") 
            sc.pp.neighbors(adata,use_rep=emb_label)
            adata.obsm['X_emb']=adata.obsm[emb_label]
            adata.obs['batch']=adata.obs['species']
            adata.obs=adata.obs[['celltype','species','batch']]
            adata.obs['batch']=adata.obs['batch'].astype('category')
            adata.obs['celltype']=adata.obs['celltype'].astype('category')
            adata_ls.obs['batch']=adata_ls.obs['batch'].astype('category')
            adata_ls.obs['celltype']=adata_ls.obs['celltype'].astype('category')
            df2=scib.metrics.metrics(adata_ls, adata, batch_key='batch', label_key='celltype', embed=emb_label, cluster_key=None, cluster_nmi=None,ari_=True, nmi_=True, nmi_method='arithmetic', nmi_dir=None, silhouette_=True, si_metric='euclidean',pcr_=True, cell_cycle_=False, organism='mouse', hvg_score_=hvg_label, isolated_labels_=True,isolated_labels_f1_=True, isolated_labels_asw_=True, n_isolated=True, graph_conn_=True,trajectory_=True, kBET_=True, lisi_graph_=True, ilisi_=True, clisi_=True, subsample=0.5, n_cores=30,type_=type_label,verbose=False)
            ari_batch,nmi_batch=compute_ARI_and_NMI_species(adata,batch_label='batch')
            df2.loc['ARI_batch']=1-ari_batch
            df2.loc['NMI_batch']=1-nmi_batch
            df2.to_csv(save_dir_output+file.replace('-basal.h5ad','_reconstruction_data_scib_output_new.csv'))
            print('metric finish!')
