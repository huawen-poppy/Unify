#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J evaluation
#SBATCH -o /ibex/scratch/projects/c2101/species_integration/logs/evaluation.%J.out
#SBATCH -e /ibex/scratch/projects/c2101/species_integration/logs/evaluation.%J.err
#SBATCH --time=6:00:00
#SBATCH --mem=200G


conda activate saturn
module load R

python scib_running_for_method_development.py --input_folder ./best_model_outputs/ \
    --processed_data_folder ../data/ \
    --target task12 \
    --save_dir ./best_model_outputs_evaluation/ \
    --file_name task12-basal.h5ad 
