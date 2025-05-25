#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J esm_llama_all_pareto2
#SBATCH -o /ibex/scratch/projects/c2101/species_integration/logs/esm_llama_all_pareto2.%J.out
#SBATCH -e /ibex/scratch/projects/c2101/species_integration/logs/esm_llama_all_pareto2.%J.err
#SBATCH --time=80:00:00
#SBATCH --mem=200G          ##SBATCH --gres=gpu:1  #SBATCH --constraint=v100

conda activate scanpy
module load R

cd /ibex/scratch/projects/c2101/species_integration/esm_llama_all_gene_aae_pareto
#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task6/tu982sye --count 20
#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task8/f76i9oeo --count 20
#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task10/4foqrfn9 --count 20
#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task7/0dcqax5y --count 20 

## for only using the BP gene set

#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task7_onlyBP/769niz5y --count 20
#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task10_onlyBP/ggcx58hz --count 20

# sweep the task10 with only BP gene set on the number of the llama clusters
#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task10_onlyBP_sweep_llama_clusters/e8mik6q6 --count 10  # batch 512

#wandb agent huawen-zhong/esmAAE_llama_all_gene_pareto_sweep_task10_allGO_sweep_llama_clusters/aspeiywp --count 20  # batch 512
#python /ibex/scratch/projects/c2101/species_integration/esm_llama_all_gene_aae_pareto/esmAAE_train_model.py --h5ad_files ../data/task10_fish.h5ad ../data/task10_fly.h5ad --species_labels fish fly --celltype_labels cell_lineage cell_lineage --output_path ./models/task10_sweep --gene_esm_embedding_path ../gene_embeddings/task10_fish_v11.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task10_fly.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/task10_fish_llama2-7B_gene_embeddings.pt ../gene_go_term/task10_fly_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

python species_specific_ASW.py
#python species-specific_GC.py