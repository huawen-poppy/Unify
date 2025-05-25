#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J esmLLAMA
#SBATCH -o /ibex/scratch/projects/c2101/species_integration/logs/esmllama.%J.out
#SBATCH -e /ibex/scratch/projects/c2101/species_integration/logs/esmllama.%J.err
#SBATCH --time=6:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100

conda activate scanpy
module load R

cd /ibex/scratch/projects/c2101/species_integration/esm_llama_all_gene_aae_pareto

# task3
#python best_esmllama_model_generation.py --h5ad_files ../data/task3_cat.h5ad ../data/task3_tiger.h5ad --species_labels cat tiger --celltype_labels NewCelltype NewCelltype --output_path ./models/task3_sweep --gene_esm_embedding_path ../gene_embeddings/task3_cat.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task3_tiger.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/task3_cat_llama2-7B_gene_embedding.pt ../gene_go_term/all_go_gene_embedding/task3_tiger_llama2-7B_gene_embedding.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

# task 7
#python best_esmllama_model_generation.py --h5ad_files ../data/task7_cat.h5ad ../data/task7_dog.h5ad --species_labels cat dog --celltype_labels NewCelltype NewCelltype --output_path ./models/task7_sweep --gene_esm_embedding_path ../gene_embeddings/task3_cat.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task7_dog.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/task7_cat_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task7_dog_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

# task 6 (only BP)
#python best_esmllama_model_generation.py --h5ad_files ../data/task6_nema.h5ad ../data/task6_sty.h5ad --species_labels nema sty --celltype_labels celltype celltype --output_path ./models/task6_sweep --gene_esm_embedding_path ../gene_embeddings/task6_nematostella.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task6_stylophora.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/task6_nema_llama2-7B_gene_embeddings.pt ../gene_go_term/task6_sty_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

# task 8
#python best_esmllama_model_generation.py --h5ad_files ../data/task8_fish.h5ad ../data/task8_frog.h5ad --species_labels fish frog --celltype_labels cell_type cell_type --output_path ./models/task8_sweep --gene_esm_embedding_path ../gene_embeddings/task8_fish_v10.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task8_frog_new.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/task8_fish_llama2-7B_gene_embeddings.pt ../gene_go_term/task8_frog_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

# task 10
#python best_esmllama_model_generation2.py --h5ad_files ../data/task10_fish.h5ad ../data/task10_fly.h5ad --species_labels fish fly --celltype_labels cell_lineage cell_lineage --output_path ./models/task10_sweep --gene_esm_embedding_path ../gene_embeddings/task10_fish_v1.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task10_fly.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/task10_fish_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task10_fly_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

# task 41 human to annotate the mouse 
#python best_esmllama_model_generation2.py --h5ad_files ../data/task41_human.h5ad ../data/task41_mouse.h5ad --species_labels human mouse --celltype_labels cell_cluster cell_cluster --output_path ./models/task41_sweep --gene_esm_embedding_path ../gene_embeddings/task4_human.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task4_mouse.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/task41_human_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task41_mouse_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True --celltype_annotation_ref 'human'


# task 40 human and mouse integration (with shared and species-specific celltypes)
#python best_esmllama_model_generation2.py --h5ad_files ../data/task40_human.h5ad ../data/task40_mouse.h5ad --species_labels human mouse --celltype_labels final_annotation final_annotation --output_path ./models/task40_sweep --gene_esm_embedding_path ../gene_embeddings/task4_human.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task4_mouse.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/task41_human_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task41_mouse_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

# multiple species integration
# task3 cat, task3 tiger, task7dog
#python /ibex/scratch/projects/c2101/species_integration/esm_llama_all_gene_aae_pareto/best_esmllama_model_generation2.py --h5ad_files ../data/task3_cat.h5ad ../data/task7_dog.h5ad ../data/task3_tiger.h5ad \
#       --species_labels cat dog tiger --celltype_labels NewCelltype NewCelltype NewCelltype --output_path ./models/task373_sweep \
#       --gene_esm_embedding_path ../gene_embeddings/task3_cat.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task7_dog.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task3_tiger.gene_symbol_to_embedding_ESM1b.pt \
#       --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/task3_cat_llama2-7B_gene_embedding.pt ../gene_go_term/all_go_gene_embedding/task7_dog_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task3_tiger_llama2-7B_gene_embedding.pt \
#       --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True


# 5 species eye datasets
#python /ibex/scratch/projects/c2101/species_integration/esm_llama_all_gene_aae_pareto/best_esmllama_model_generation2.py --h5ad_files ../data/aqueous_atlas/MF.h5ad ../data/aqueous_atlas/MM.h5ad ../data/aqueous_atlas/human.h5ad ../data/aqueous_atlas/mouse.h5ad ../data/aqueous_atlas/pig.h5ad \
#       --species_labels MF MM human mouse pig --celltype_labels cell_type cell_type cell_type cell_type cell_type --output_path ./models/task55_sweep \
#       --gene_esm_embedding_path ../../SATURN/data/embedding/task19_MF.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task20_MM.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task4_human.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task4_mouse.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task24_pig.gene_symbol_to_embedding_ESM2.pt \
#       --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/mf_emapper_llama2-7B_all_go_description.pt ../gene_go_term/all_go_gene_embedding/mm_emapper_llama2-7B_all_go_description.pt ../gene_go_term/all_go_gene_embedding/task41_human_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task41_mouse_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/pig_emapper_llama2-7B_all_go_description.pt \
#       --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## celltype tree 
#python /ibex/scratch/projects/c2101/species_integration/esm_llama_all_gene_aae_pareto/best_esmllama_model_generation2.py --h5ad_files ../data/celltype_tree/celegan.h5ad ../data/celltype_tree/ciona.h5ad ../data/celltype_tree/fish.h5ad ../data/celltype_tree/fly.h5ad ../data/celltype_tree/human.h5ad ../data/celltype_tree/mouse.h5ad ../data/celltype_tree/schmidtea.h5ad \
#       --species_labels celegan ciona fish fly human mouse schmidtea --celltype_labels cell_lineage cell_lineage cell_lineage cell_lineage cell_lineage cell_lineage cell_lineage --output_path ./models/task77_sweep \
#       --gene_esm_embedding_path ../../SATURN/data/embedding/task14_celegan.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task12_ciona.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task8_fish_v10.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task10_fly.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task4_human.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task4_mouse.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task16_schmidtea.gene_symbol_to_embedding_ESM2.pt \
#       --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/celegan_emapper_llama2-7B_all_go_description.pt ../gene_go_term/all_go_gene_embedding/ciona_emapper_llama2-7B_all_go_description.pt ../gene_go_term/all_go_gene_embedding/task10_fish_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task10_fly_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task41_human_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task41_mouse_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/schmid_emapper_llama2-7B_all_go_description.pt \
#       --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## chimpanzee and human ipsc neurons
#python /ibex/scratch/projects/c2101/species_integration/esm_llama_all_gene_aae_pareto/best_esmllama_model_generation2.py --h5ad_files ../data/chimpanzee_brain/chimp.h5ad ../data/chimpanzee_brain/human.h5ad \
#       --species_labels chimp human --celltype_labels celltype celltype --output_path ./models/task69_sweep \
#       --gene_esm_embedding_path ../../SATURN/data/embedding/panTro5.gene_symbol_to_embedding_ESM2.pt ../../SATURN/data/embedding/task4_human.gene_symbol_to_embedding_ESM2.pt \
#       --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/chimpan_emapper_llama2-7B_all_go_description.pt ../gene_go_term/all_go_gene_embedding/task41_human_llama2-7B_gene_embeddings.pt \
#       --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## task88, for cell type annotation, from zebrafish to frog
#python best_esmllama_model_generation2.py --h5ad_files ../data/task8_fish.h5ad ../data/task88_frog.h5ad --species_labels fish frog --celltype_labels cell_type leiden --output_path ./models/task88_sweep --gene_esm_embedding_path ../gene_embeddings/task8_fish_v10.gene_symbol_to_embedding_ESM1b.pt ../gene_embeddings/task8_frog_new.gene_symbol_to_embedding_ESM1b.pt --gene_llama_embedding_path ../gene_go_term/task8_fish_llama2-7B_gene_embeddings.pt ../gene_go_term/task8_frog_llama2-7B_gene_embeddings.pt --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## task9, for the low quality sequencing depth dataset evaluation
#python best_esmllama_model_generation2.py --h5ad_files /ibex/project/c2101/SATURN/data/origin_h5ad/task9_ant.h5ad /ibex/project/c2101/SATURN/data/origin_h5ad/task9_mouse.h5ad \
#    --species_labels ant mouse \
#    --celltype_labels celltype celltype \
#    --output_path ./models/task9_sweep \
#    --gene_esm_embedding_path /ibex/scratch/projects/c2101/SATURN/data/embedding/task9_ant.gene_symbol_to_embedding_ESM1b.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/task4_mouse.gene_symbol_to_embedding_ESM1b.pt \
#    --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/ant_gene_to_all_go_descriptions.pt ../gene_go_term/all_go_gene_embedding/task41_mouse_llama2-7B_gene_embeddings.pt \
#    --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## task9-1, for the low quality sequencing depth dataset evaluation
#python best_esmllama_model_generation2.py --h5ad_files /ibex/project/c2101/SATURN/data/origin_h5ad/task9-1_ant.h5ad /ibex/project/c2101/SATURN/data/origin_h5ad/task9-1_mouse2_adult.h5ad \
#    --species_labels ant mouse \
#    --celltype_labels celltype celltype \
#    --output_path ./models/task9-1_sweep \
#    --gene_esm_embedding_path /ibex/scratch/projects/c2101/SATURN/data/embedding/task9_ant.gene_symbol_to_embedding_ESM1b.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/task4_mouse.gene_symbol_to_embedding_ESM1b.pt\
#    --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/ant_gene_to_all_go_descriptions.pt ../gene_go_term/all_go_gene_embedding/task41_mouse_llama2-7B_gene_embeddings.pt \
#    --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## task27 pig and MM for the species-specific cell type 
#python best_esmllama_model_generation2.py --h5ad_files /ibex/project/c2101/SATURN/data/origin_h5ad/task27_MM.h5ad /ibex/project/c2101/SATURN/data/origin_h5ad/task27_pig.h5ad \
#    --species_labels MM pig \
#    --celltype_labels cell_type cell_type \
#    --output_path ./models/task27_sweep \
#    --gene_esm_embedding_path /ibex/scratch/projects/c2101/SATURN/data/embedding/task20_MM.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/task24_pig.gene_symbol_to_embedding_ESM2.pt \
#    --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/mm_emapper_llama2-7B_all_go_description.pt ../gene_go_term/all_go_gene_embedding/pig_emapper_llama2-7B_all_go_description.pt \
#    --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## task12 ciona and nematostella for the species-specific cell type
#python best_esmllama_model_generation2.py --h5ad_files /ibex/project/c2101/SATURN/data/origin_h5ad/task12_ciona.h5ad /ibex/project/c2101/SATURN/data/origin_h5ad/task12_nematostella.h5ad \
#    --species_labels ciona nemato \
#    --celltype_labels cell_lineage cell_lineage \
#    --output_path ./models/task12_sweep \
#    --gene_esm_embedding_path /ibex/scratch/projects/c2101/SATURN/data/embedding/task12_ciona.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/nematostella.gene_symbol_to_embedding_ESM2.pt \
#    --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/ciona_emapper_llama2-7B_all_go_description.pt ../gene_go_term/all_go_gene_embedding/task6_nema_llama2-7B_gene_embeddings.pt \
#    --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## cytokine perturbation prediction task99
#python best_esmllama_model_generation2.py --h5ad_files /ibex/project/c2101/species_integration/data/perturbation/human_monocyte_pbs_subset_pp.h5ad /ibex/project/c2101/species_integration/data/perturbation/mouse_monocyte_pbs_ifn2_pp.h5ad \
#    --species_labels human mouse \
#    --celltype_labels cytokine cytokine \
#   --output_path ./models/task99_pp_data_decoder_without_relu_sweep_monocyte_ifn2 \
#    --gene_esm_embedding_path /ibex/scratch/projects/c2101/SATURN/data/embedding/task4_human.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/task4_mouse.gene_symbol_to_embedding_ESM2.pt \
#    --gene_llama_embedding_path ../gene_go_term/all_go_gene_embedding/task41_human_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task41_mouse_llama2-7B_gene_embeddings.pt \
#   --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

## for the cnidarian atlas integration task101
#python best_esmllama_model_generation2.py --h5ad_files /ibex/scratch/projects/c2101/Acropora_sc_analysis/multi_species/data/origin_data/ahem_origin.h5ad /ibex/scratch/projects/c2101/Acropora_sc_analysis/multi_species/data/origin_data/aiptasia_apo_sym.h5ad /ibex/scratch/projects/c2101/Acropora_sc_analysis/multi_species/data/hydra_new.h5ad /ibex/scratch/projects/c2101/Acropora_sc_analysis/multi_species/data/nemato_new.h5ad /ibex/scratch/projects/c2101/Acropora_sc_analysis/multi_species/data/sty_new.h5ad /ibex/scratch/projects/c2101/Acropora_sc_analysis/multi_species/data/xen_new.h5ad \
#    --species_labels ahem aiptasia hydra nematostella stylo xenia \
#    --celltype_labels cell_type cell_type cell_type cell_type cell_type cell_type \
#    --output_path ./models/task101_sweep \
#    --gene_esm_embedding_path /ibex/scratch/projects/c2101/SATURN/data/embedding/ahem.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/aiptasia.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/hydra.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/nematostella.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/stylophora.gene_symbol_to_embedding_ESM2.pt /ibex/scratch/projects/c2101/SATURN/data/embedding/xenia.gene_symbol_to_embedding_ESM2.pt  \
#    --gene_llama_embedding_path  ../gene_go_term/all_go_gene_embedding/ahem.emapper_gene_to_all_go_descriptions.pt ../gene_go_term/all_go_gene_embedding/aiptasia.emapper_gene_to_all_go_descriptions.pt ../gene_go_term/all_go_gene_embedding/hydra.emapper_gene_to_all_go_descriptions.pt ../gene_go_term/all_go_gene_embedding/task6_nema_llama2-7B_gene_embeddings.pt ../gene_go_term/all_go_gene_embedding/task6_sty_llama2-7B_gene_embeddings_allGO.pt  ../gene_go_term/all_go_gene_embedding/xenia.emapper_gene_to_all_go_descriptions.pt \
#    --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True

# drug perturbation mouse and pig 
python best_esmllama_model_generation2.py --h5ad_files $1 $2 \
    --species_labels $3 $4 \
    --celltype_labels condition condition \
    --output_path $5 \
    --gene_esm_embedding_path $6 $7 \
    --gene_llama_embedding_path $8 $9 \
    --num_esm_macrogene 2000 --num_llama_macrogene 2000 --highly_variable_genes 8000 --evaluate_emb True