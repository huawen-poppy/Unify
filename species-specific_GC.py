import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
import argparse
import re
import scanpy as sc

def graph_connectivity_specific_cell_type(adata, celltype_names):
    if "neighbors" not in adata.uns:
        raise KeyError("Please compute the knn graph before running this function"
                )
    clust_res = []
    for celltype_name in celltype_names:
        adata_sub = adata[adata.obs['cell_type']==celltype_name]
        _, labels = connected_components(
                adata_sub.obsp['connectivities'], connection="strong")
        tab = pd.value_counts(labels)
        clust_res.append(tab.max()/sum(tab))
    return np.mean(clust_res)

target_files=['task3','task7','task27','task12','task8']

for target in target_files:
    adata=sc.read_h5ad('/ibex/project/c2101/species_integration/esm_llama_all_gene_aae_pareto/best_model_outputs/'+target+'-basal.h5ad')
    adata.obs['cell_type']=adata.obs['celltype']
    adata.obs['batch']=adata.obs['species']
    sc.pp.neighbors(adata, use_rep='X_esm')
    species_names=np.unique(adata.obs['batch'])
    sp_label1=np.unique(adata.obs[adata.obs['batch']==species_names[0]]['cell_type'])
    sp_label2=np.unique(adata.obs[adata.obs['batch']==species_names[1]]['cell_type'])
    diff=set(sp_label2).symmetric_difference(set(sp_label1))
    celltype_names=list(diff)

    gc_species_specific_celltype=graph_connectivity_specific_cell_type(adata,celltype_names)
    gc_df=pd.DataFrame([gc_species_specific_celltype],columns=['gc_species_specific_celltype_value'])
    gc_df.to_csv('/ibex/project/c2101/species_integration/figure2_benchmark/'+target+'_galaxy_GC_species_specific_celltype.csv')