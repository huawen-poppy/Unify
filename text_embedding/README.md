# Gene Description Embeddings with GO Terms

This folder demonstrates how to generate **gene description embeddings** using functional annotations such as **GO terms**.  

The human GO term information is downloaded from UniProt (reviewed entries only):  
👉 [Download link](https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Cgene_names%2Cgo&format=tsv&query=%28organism_name%3Ahuman%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29)  

A similar procedure is applied for **mouse GO term information**.

---

## Notes
1. We use the **LLaMA-2 model** as an example to generate gene description embeddings, but **any LLM** can be applied.  
2. The **gene ontology (GO) term** is used as the description, but other functional annotations (e.g., **KEGG pathways**, **Reactome pathways**) can also be used.

---

## Methods for Obtaining Gene Descriptions

### A. From NCBI Descriptions
- For non-human species, we use the [`mygene` package](https://docs.mygene.info/en/latest/doc/data.html#data-sources) to extract **gene IDs** from multiple databases (NCBI, UniProt, Ensembl, etc.).  
- `mygene` supports all species in NCBI, accepting either **species name** or **taxonomic ID**.  

**Pipeline:**
1. Generate a **gene → gene ID dictionary** (JSON).  
2. Use this dictionary to extract gene functional descriptions from the **NCBI website** (can be time-consuming).  
3. Run the LLaMA model to generate description embeddings.  

**Disadvantages:**
- Downloading time can be long and network interruptions may occur.  
- For **non-model species**, functional descriptions may be missing even if the gene exists in model organisms.  

---

### B. From `gene_ontology.gaf` Files
1. Download GO term annotations:  
   - [GO Ontology](https://geneontology.org/docs/download-ontology/)  
   - [GO Annotations](https://geneontology.org/docs/download-go-annotations/)  
   - [EBI GOA Proteomes](https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes/)  
   - Alternative: [QuickGO](https://www.ebi.ac.uk/QuickGO/annotations?taxonId=9544&taxonUsage=descendants)  
   - Alternative: [UniProt](https://www.uniprot.org/)  

2. Run the `generate_gene_description.py` script to:  
   - Create the **gene → description mapping** (JSON).  
   - Generate **gene → embedding files**.  

---

### C. From Protein Sequence Files
1. Use [eggNOG-mapper](http://eggnog-mapper.embl.de/submit_job) (online or local installation).  
2. Generate **protein sequence annotations with GO terms**.  
3. Obtain **protein → GO description** and embedding files.  
4. Adjust/match **protein IDs → gene IDs** in the embedding file.  

---

## Summary
This README provides multiple workflows for generating **gene functional descriptions and embeddings**:
- Using **NCBI descriptions** (via `mygene`)  
- Using **GO annotation files**  
- Using **protein sequence annotation tools**  

Each approach has trade-offs in terms of completeness, ease of use, and applicability to non-model organisms.  