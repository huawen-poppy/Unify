# **Galaxy: Deciphering Cellular Evolution with Universal Multimodal Embeddings**

This repository contains the code for training and generating results using the Galaxy model.

## **Overview**
![Architecture Diagram](images/model.png)

**Galaxy** is a **transfer learning framework** designed to integrate single-cell RNA sequencing (**scRNA-seq**) data **across species** — even those separated by hundreds of millions of years.  
Instead of relying solely on one-to-one orthologs, Galaxy creates **functionally coherent multi-modal macrogenes** that **transcend species boundaries** by integrating:  

- **Gene expression data** (RNA-seq)  
- **Protein language model embeddings**  
- **General-purpose large language model embeddings**  

This approach allows Galaxy to:  
- ✅ Correct **technical batch effects** while preserving conserved biology  
- ✅ Capture **functionally convergent gene programs** beyond strict orthology  
- ✅ Predict **cross-species perturbation responses** (e.g., mouse → human)  
- ✅ Build **multi-species evolutionary trees** of cell types with higher accuracy  

By uniting molecular and computational insights, Galaxy opens a **new avenue for comparative single-cell genomics**, enabling biological discovery across vast evolutionary distances.  

