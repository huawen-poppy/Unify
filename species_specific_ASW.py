import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import time
from datetime import timedelta
import scanpy as sc
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
print(sc.logging.print_versions())
import os
dirname = os.getcwd()
print(dirname)
from sklearn.metrics import silhouette_score
import random
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, silhouette_samples
import anndata as ad

def silhouette_coeff_ASW_species_specific_celltype(adata, celltype_names=[],method_use='raw',save_dir='', task_name='', percent_extract=0.8):
    random.seed(0)
    asw_fscore = []
    asw_bn = []
    asw_bn_sub = []
    asw_ctn = []
    iters = []
    for celltype_name in celltype_names:
        i=0
        iters.append('iteration_'+str(i+1))
        mask = adata.obs['cell_type'] == celltype_name

        rand_cidx = list(adata.obs.index.get_indexer(mask[mask].index))

        asw_celltype = silhouette_samples(adata.X, adata.obs['cell_type'])
        min_val = -1
        max_val = 1
        asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)

        asw_ctn.append(np.mean(asw_celltype_norm[rand_cidx]))

    df = pd.DataFrame({'asw_celltype_norm': asw_ctn,'method_use':np.repeat(method_use, len(asw_ctn))})
    df_mean=df.mean()
    df=df.append(df_mean,ignore_index=True)
    df.to_csv(save_dir + task_name + "_"+method_use+'_ASW_metric_species_sepcific_celltype.csv')
    print('Save output of pca in: ',save_dir)
    return df



def run_asw_species_specific_celltype(adata,method='galaxy',save_dir='../../figure2_benchmark/',task_name=''):
    adata_new=ad.AnnData(adata.obsm['X_esm'])
    adata_new.obs=adata.obs
    adata_new.obs['batch']=adata.obs['species']
    adata_new.obs['cell_type']=adata.obs['celltype']
    species_names=np.unique(adata_new.obs['batch'])

    sp_label1=np.unique(adata_new.obs[adata_new.obs['batch']==species_names[0]]['cell_type'])
    sp_label2=np.unique(adata_new.obs[adata_new.obs['batch']==species_names[1]]['cell_type'])

    diff=set(sp_label2).symmetric_difference(set(sp_label1))
    celltype_names=list(diff)
    AWS_species_specific=silhouette_coeff_ASW_species_specific_celltype(adata_new, celltype_names=celltype_names,method_use=method,save_dir=save_dir, task_name=task_name, percent_extract=0.8)


adata=sc.read_h5ad('/ibex/project/c2101/species_integration/esm_llama_all_gene_aae_pareto/best_model_outputs/task8-basal.h5ad')
run_asw_species_specific_celltype(adata,task_name='task8')