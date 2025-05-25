#!/bin/bash 

data_dir="/ibex/project/c2101/species_integration/data/perturbation/lps/"
h5ad_files=($(find "$data_dir" -maxdepth 1 -name "*.h5ad"))

output_root="./models/"

gene_esm_root="/ibex/scratch/projects/c2101/SATURN/data/embedding"
gene_llama_root="/ibex/project/c2101/species_integration/gene_go_term/all_go_gene_embedding"


processed_pairs=()

for i in "${!h5ad_files[@]}"; do
  file1="${h5ad_files[$i]}"
  filename1=$(basename "$file1")

  if [[ "$filename1" =~ ^(mouse|pig)([0-9]+)_control_lps2\.h5ad$ ]]; then
    species1="${BASH_REMATCH[1]}"
    sample_id1="${BASH_REMATCH[2]}"

    for j in "${!h5ad_files[@]}"; do
      if [[ "$i" -ne "$j" ]]; then
        file2="${h5ad_files[$j]}"
        filename2=$(basename "$file2")
        if [[ "$filename2" =~ ^(mouse|pig)([0-9]+)_control\.h5ad$ ]]; then
          species2="${BASH_REMATCH[1]}"
          sample_id2="${BASH_REMATCH[2]}"

          # Check if species are different and if this pair hasn't been processed
          if [[ "$species1" != "$species2" ]]; then
            pair_identifier="${species1}_${sample_id1}_vs_${species2}_${sample_id2}"
            processed=false
            for processed_pair in "${processed_pairs[@]}"; do
              if [[ "$processed_pair" == "$pair_identifier" ]]; then
                processed=true
                break
              fi
            done

            if [[ "$processed" == false ]]; then
              output_path="${output_root}/${species1}${sample_id1}_predict_${species2}${sample_id2}"

              if [[ "$species1" == "mouse" ]]; then
                gene_esm_path1="${gene_esm_root}/task4_mouse.gene_symbol_to_embedding_ESM2.pt"
              elif [[ "$species1" == "pig" ]]; then
                gene_esm_path1="${gene_esm_root}/task24_pig.gene_symbol_to_embedding_ESM2.pt"
              else
                echo "Error: Unknown species '$species1' for ESM embedding."
                continue
              fi

              if [[ "$species2" == "mouse" ]]; then
                gene_esm_path2="${gene_esm_root}/task4_mouse.gene_symbol_to_embedding_ESM2.pt"
              elif [[ "$species2" == "pig" ]]; then
                gene_esm_path2="${gene_esm_root}/task24_pig.gene_symbol_to_embedding_ESM2.pt"
              else
                echo "Error: Unknown species '$species2' for ESM embedding."
                continue
              fi

              if [[ "$species1" == "mouse" ]]; then
                gene_llama_path1="${gene_llama_root}/task41_mouse_llama2-7B_gene_embeddings.pt"
              elif [[ "$species1" == "pig" ]]; then
                gene_llama_path1="${gene_llama_root}/pig_emapper_llama2-7B_all_go_description.pt"
              else
                echo "Error: Unknown species '$species1' for Llama embedding."
                continue
              fi

              if [[ "$species2" == "mouse" ]]; then
                gene_llama_path2="${gene_llama_root}/task41_mouse_llama2-7B_gene_embeddings.pt"
              elif [[ "$species2" == "pig" ]]; then
                gene_llama_path2="${gene_llama_root}/pig_emapper_llama2-7B_all_go_description.pt"
              else
                echo "Error: Unknown species '$species2' for Llama embedding."
                continue
              fi

              echo "Submitting job for: $filename1 and $filename2"
              echo "$file1" "$file2" "$species1" "$species2" "$output_path" "$gene_esm_path1" "$gene_esm_path2" "$gene_llama_path1" "$gene_llama_path2"
              sbatch submit_best_model.slurm "$file1" "$file2" "$species1" "$species2" "$output_path" "$gene_esm_path1" "$gene_esm_path2" "$gene_llama_path1" "$gene_llama_path2"
              #processed_pairs+=("$pair_identifier")
            fi
          fi
        fi
      fi
    done
  fi
done
