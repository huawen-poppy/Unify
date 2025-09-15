import pickle
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import scanpy as sc
import numpy as np
from tqdm import tqdm
import torch
import os
import pandas as pd
import obonet

# Set environment variable to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# Check if CUDA is available and set device(s)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs for training.')

def main():
    parser = argparse.ArgumentParser(description='Generate the gene go term embedding from the llama model')
    parser.add_argument('--input_gaf', type=str, help='target species go term file')
    parser.add_argument('--output_file_prefix', type=str, help='output file name')
    parser.add_argument('--whether_subset_bp', type=str, help='whether subset the go term to biological process')

    args = parser.parse_args()
    gaf_file=args.input_gaf
    output_file_prefix=args.output_file_prefix
    whether_subset_bp=args.whether_subset_bp
    if whether_subset_bp=='True':
        whether_subset_bp=True
    else:
        whether_subset_bp=False
    print(whether_subset_bp)
    # Load the file into a DataFrame
    columns = [
        "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", 
        "GO_ID", "DB_Reference", "Evidence_Code", "With_or_From", 
        "Aspect", "DB_Object_Name", "DB_Object_Synonym", 
        "DB_Object_Type", "Taxon", "Date", "Assigned_By", "Annotation_Extension", "Gene_Product_Form_ID"
    ]

    data = pd.read_csv(gaf_file, sep="\t", comment="!", names=columns)

    if whether_subset_bp:
        bp_data = data[data['Aspect']=='P']
        final_saved_name='_gene_to_bp_go_descriptions.pt'
        json_saved_name='_gene_to_bp_go_descriptions.json'
    else:
        bp_data = data
        final_saved_name='_gene_to_all_go_descriptions.pt'
        json_saved_name='_gene_to_all_go_descriptions.json'
        
    obo_file = "go-basic.obo"
    graph = obonet.read_obo(obo_file)

    # Step 2: Extract GO descriptions for biological processes
    go_descriptions = {}
    for go_id, node_data in graph.nodes(data=True):
        namespace = node_data.get("namespace", "")

        # Get description if available
        desc = node_data.get("name", "No description available")
        if desc == "No description available":
            continue  # Skip terms without descriptions
        # Handle subsetting logic
        if whether_subset_bp:
            # Only keep biological process terms
            if namespace == "biological_process":
                go_descriptions[go_id] = desc
        else:
            # Keep all terms regardless of namespace
            go_descriptions[go_id] = desc
        '''
        if whether_subset_bp:
            if data.get("namespace") == "biological_process":
                go_descriptions[go_id] = data.get("name", "No description available")
        else:
            go_descriptions[go_id] = data.get("name", "No description available")
        '''
    # Create the gene-to-description mapping
    gene_to_go_desc = {}
    unwanted_descriptions = {'molecular_function', 'biological_process', 'cellular_component'}
    for _, row in bp_data.iterrows():
        gene = row["DB_Object_Symbol"]
        go_id = row["GO_ID"]
        description = go_descriptions.get(go_id, "No description available")

        # Skip invalid or unwanted descriptions
        if not description or description in unwanted_descriptions:
            continue
        if gene not in gene_to_go_desc:
            gene_to_go_desc[gene] = []
        if description not in gene_to_go_desc[gene]:  # Avoid duplicates
            gene_to_go_desc[gene].append(description)

    # Save as JSON
    output_file = output_file_prefix+json_saved_name
    with open(output_file, "w") as f:
        json.dump(gene_to_go_desc, f, indent=4)
    
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf",cache_dir='./')
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf",cache_dir='./')
    ## generate the embedding for each gene 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for data parallelism.")
        model = torch.nn.DistributedDataParallel(model)

    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Loop over genes and descriptions
    target_gene_embedding = {}
    for gene, description in tqdm(gene_to_go_desc.items()):
        with torch.no_grad():
            if len(description) == 1:
                inputs = tokenizer(description[0], return_tensors="pt").to(device)
                last_hidden_state_temp = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                last_hidden_state = last_hidden_state_temp.cpu().detach().numpy()[0][-1]
            else:
                all_embeddings = []
                for desc in description:
                    inputs = tokenizer(desc, return_tensors="pt").to(device)
                    last_hidden_state_temp = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                    all_embeddings.append(last_hidden_state_temp.cpu().detach().numpy()[0][-1])
                stacked_embedding = np.vstack(all_embeddings)
                last_hidden_state = np.mean(stacked_embedding, axis=0)

            target_gene_embedding[gene] = torch.from_numpy(np.array(last_hidden_state))

        # Clear memory
        del last_hidden_state_temp
        torch.cuda.empty_cache()

    # Save the embeddings
    save_file_name=output_file_prefix+final_saved_name
    torch.save(target_gene_embedding, save_file_name)

main()
