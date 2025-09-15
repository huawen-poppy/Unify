from __future__ import annotations
import os, copy, json, argparse, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import scanpy as sc
import anndata as ad
from glob import glob

from models import Decoder, Discriminator, Discriminator_celltype, Encoder
from macrogene_initialize import macrogene_initialization, load_gene_embeddings_adata
from dataset import SingleCellDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

# Only import the solver; we provide our own gradient_normalizers
from min_norm_solvers import MinNormSolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Repro & small helpers
# -----------------------------
def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, encoder, decoder, epoch, metrics: dict):
    payload = {
        "epoch": int(epoch),
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path, encoder, decoder, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    return ckpt


def is_better(curr, best, mode="max"):
    if best is None:
        return True
    return (curr > best) if mode == "max" else (curr < best)


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def compute_ARI_and_NMI_celltype(y_true, y_feat):
    """
    y_true: true labels
    y_feat: features to cluster
    """
    n = 20
    resolution = [2 * x / n for x in range(1, n + 1)]
    _adata = ad.AnnData(X=y_feat)
    _adata.obs['celltype'] = y_true
    best_ari = 0.0
    for res in resolution:
        sc.tl.pca(_adata)
        _adata.obsm['X_emb'] = _adata.X
        sc.pp.neighbors(_adata, use_rep='X_emb')
        sc.tl.louvain(_adata, resolution=res, key_added='louvain')
        ari = adjusted_rand_score(y_true, _adata.obs['louvain'])
        if ari > best_ari:
            best_ari = ari
        del _adata.obs['louvain']
    return float(best_ari)


def evaluation_model_embed(encoder: nn.Module, dataset: SingleCellDataset) -> float:
    encoder.eval()
    gene_input = dataset.trans_profiles.to(device).float()
    with torch.no_grad():
        latent = encoder(gene_input)
    celltype_idx = dataset.celltype_id_embedding_all
    ari = compute_ARI_and_NMI_celltype(celltype_idx, latent.cpu().numpy())
    return float(ari)


# -----------------------------
# The gradient normalizer (kept)
# -----------------------------
def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in range(len(grads)):
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in range(len(grads)):
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in range(len(grads)):
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in range(len(grads)):
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn


# -----------------------------
# Post-training generation
# -----------------------------
def generate_outputs_from_models(encoder, decoder, dataset, outdir: Path, epoch_used: int):
    encoder.eval(); decoder.eval()
    gene_input = dataset.trans_profiles.to(device).float()
    with torch.no_grad():
        latent = encoder(gene_input)
    celltype_idx = dataset.celltype_id_embedding_all
    ari_embed = compute_ARI_and_NMI_celltype(celltype_idx, latent.cpu().numpy())
    np.save(outdir / "latent.npy", latent.cpu().numpy())
    with open(outdir / "inference_summary.json", "w") as f:
        json.dump({
            "epoch": int(epoch_used),
            "ARI_embeddings": float(ari_embed),
            "checkpoint": str(outdir / "best.pt")
        }, f, indent=2)
    print(f"[E2E] Saved latent to {outdir/'latent.npy'} and metrics to {outdir/'inference_summary.json'}")



# -------------------------------------------
# Final target file generator after training 
# -------------------------------------------
def merge_species_and_attach_latent(outdir: Path, sorted_species_names: list[str]):
    # 1) Load latent
    latent = np.load(outdir / "latent.npy")
    
    # 2) Load all per-species macrogene adatas in the SAME order used to build the dataset
    adatas = []
    for sp in sorted_species_names:
        fp = outdir / f"{sp}_macrogene_adata.h5ad"
        if not fp.exists():
            raise FileNotFoundError(f"Missing macrogene file: {fp}")
        _a = ad.read_h5ad(fp)
        _a.obs["species"] = sp  # keep a species tag
        adatas.append(_a)

    # 3) Concatenate by rows (cells)
    merged = ad.concat(adatas, axis=0, join="outer", label="species_key", keys=sorted_species_names, index_unique="-")

    # 4) Sanity check: latent must match number of cells
    if latent.shape[0] != merged.n_obs:
        raise ValueError(f"latent rows ({latent.shape[0]}) != merged cells ({merged.n_obs}). "
                         "Ensure the per-species ordering matches dataset construction.")

    # 5) Attach embedding
    merged.obsm["X_unify"] = latent

    # 6) Save
    merged_fp = outdir / "macrogene_merged_with_unify.h5ad"
    merged.write_h5ad(merged_fp)
    print(f"[E2E] Wrote merged macrogene AnnData with integrated emnbeddings to {merged_fp}")

# -----------------------------
# Main training (end-to-end)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train the supervised AAE with macrogene layer — end-to-end (no W&B).')
    parser.add_argument('--h5ad_files', nargs='+', required=True)
    parser.add_argument('--species_labels', nargs='+', required=True)
    parser.add_argument('--celltype_labels', nargs='+', required=True)
    parser.add_argument('--gene_esm_embedding_path', nargs='+', required=True)
    parser.add_argument('--gene_llama_embedding_path', nargs='+', required=True)
    parser.add_argument('--num_esm_macrogene', type=int, default=2000)
    parser.add_argument('--num_llama_macrogene', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--highly_variable_genes', type=int, default=8000)
    parser.add_argument('--batch_labels', type=str, default=None)
    parser.add_argument('--evaluate_emb', type=str, default="True")
    parser.add_argument('--celltype_annotation_ref', type=str, default=None,
                        help='If not None, use reference species dataset to annotate another one.')

    # moved hyperparams from W&B -> CLI (defaults = your previous values)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--grad_normalized_type', type=str, default='l2')
    parser.add_argument('--hidden_size_adversary_species', type=int, default=256)
    parser.add_argument('--hidden_size_adversary_celltype', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--lr_discriminator_celltype', type=float, default=0.00018364960390400008)
    parser.add_argument('--lr_discriminator_species', type=float, default=0.00015139547698955852)
    parser.add_argument('--lr_encoder', type=float, default=0.0008269200693451001)
    parser.add_argument('--lr_generator', type=float, default=0.00010667566651855372)
    parser.add_argument('--macrogene_decoder_dropout_ratio', type=float, default=0.3)
    parser.add_argument('--macrogene_decoder_hidden_size', type=int, default=256)
    parser.add_argument('--macrogene_encoder_dropout_ratio', type=float, default=0.2)
    parser.add_argument('--macrogene_encoder_hidden_size', type=int, default=256)
    parser.add_argument('--train_epochs', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=5, help='Compute ARI every N epochs (for best selection)')
    args = parser.parse_args()

    # Seeds & paths
    set_seed(args.seed)
    outdir = Path(args.output_path)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best.pt"
    best_metric = None
    best_epoch = None
    monitor_mode = "max"   # maximize ARI
    eval_every = max(1, int(args.eval_every))

    # Resolve flags
    evaluate_emb = (args.evaluate_emb == "True")
    celltype_annotation_ref = args.celltype_annotation_ref

    # -----------------------------
    # STEP 1: load adata + gene embeddings
    # -----------------------------
    species_to_adata = {args.species_labels[i]: sc.read_h5ad(args.h5ad_files[i]) for i in range(len(args.species_labels))}
    species_to_gene_esm_embeddings_path = {args.species_labels[i]: args.gene_esm_embedding_path[i] for i in range(len(args.species_labels))}
    species_to_gene_llama_embeddings_path = {args.species_labels[i]: args.gene_llama_embedding_path[i] for i in range(len(args.species_labels))}

    species_to_adata_esm = copy.deepcopy(species_to_adata)
    species_to_adata_llama = copy.deepcopy(species_to_adata)

    species_to_gene_esm_embeddings = {}
    species_to_gene_llama_embeddings = {}

    for species, adata in species_to_adata_esm.items():
        adata, species_gene_esm_embeddings = load_gene_embeddings_adata(adata, species=[species], embedding_path=species_to_gene_esm_embeddings_path[species])
        species_to_gene_esm_embeddings.update(species_gene_esm_embeddings)
        species_to_adata_esm[species] = adata
        print('Subsetting', species, 'with ESM embeddings')

    for species, adata in species_to_adata_llama.items():
        adata, species_gene_llama_embeddings = load_gene_embeddings_adata(adata, species=[species], embedding_path=species_to_gene_llama_embeddings_path[species])
        species_to_gene_llama_embeddings.update(species_gene_llama_embeddings)
        species_to_adata_llama[species] = adata
        print('Subsetting', species, 'with LLaMA embeddings')

    sorted_species_names = sorted(list(species_to_adata.keys()))
    gene_amount_before_hvg_esm = min([v.shape[1] for v in species_to_adata_esm.values()])
    gene_amount_before_hvg_llama = min([v.shape[1] for v in species_to_adata_llama.values()])
    high_variable_genes_esm = min(args.highly_variable_genes, gene_amount_before_hvg_esm)
    high_variable_genes_llama = min(args.highly_variable_genes, gene_amount_before_hvg_llama)

    # -----------------------------
    # STEP 2: HVGs and per-species gene counts
    # -----------------------------
    species_to_gene_idx_hvg_esm = {}
    species_to_gene_idx_hvg_llama = {}
    llama_genes_per_species = {}
    esm_genes_per_species = {}

    ct = 0
    for species in sorted_species_names:
        adata = species_to_adata_esm[species]
        sc.pp.highly_variable_genes(adata, n_top_genes=high_variable_genes_esm, flavor='seurat_v3')
        hvg_index = adata.var['highly_variable']
        species_to_adata_esm[species] = adata[:, hvg_index]
        species_to_gene_esm_embeddings[species] = species_to_gene_esm_embeddings[species][hvg_index]
        species_to_gene_idx_hvg_esm[species] = (ct, ct + species_to_gene_esm_embeddings[species].shape[0])
        ct += species_to_gene_esm_embeddings[species].shape[0]
        esm_genes_per_species[species] = species_to_adata_esm[species].shape[1]

    ct = 0
    for species in sorted_species_names:
        adata = species_to_adata_llama[species]
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=high_variable_genes_llama)
        hvg_index = adata.var['highly_variable']
        species_to_adata_llama[species] = adata[:, hvg_index]
        species_to_gene_llama_embeddings[species] = species_to_gene_llama_embeddings[species][hvg_index]
        species_to_gene_idx_hvg_llama[species] = (ct, ct + species_to_gene_llama_embeddings[species].shape[0])
        ct += species_to_gene_llama_embeddings[species].shape[0]
        llama_genes_per_species[species] = species_to_adata_llama[species].shape[1]

    # -----------------------------
    # STEP 3: concatenate names + embeddings
    # -----------------------------
    all_gene_names_esm = []
    for species in sorted_species_names:
        adata = species_to_adata_esm[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        all_gene_names_esm += list(species_str.str.cat(gene_names, sep='_'))

    all_gene_names_llama = []
    for species in sorted_species_names:
        adata = species_to_adata_llama[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        all_gene_names_llama += list(species_str.str.cat(gene_names, sep='_'))

    all_gene_esm_embeddings = torch.cat([species_to_gene_esm_embeddings[s] for s in sorted_species_names], dim=0)
    all_gene_llama_embeddings = torch.cat([species_to_gene_llama_embeddings[s] for s in sorted_species_names], dim=0)

    # -----------------------------
    # STEP 4: macrogene init
    # -----------------------------
    esm_macrogene_amount = args.num_esm_macrogene
    llama_macrogene_amount = args.num_llama_macrogene

    esm_macrogene_weights = macrogene_initialization(all_gene_esm_embeddings, all_gene_names_esm,
                                                     num_macrogene=esm_macrogene_amount, normalize=False, seed=0)
    llama_macrogene_weights = macrogene_initialization(all_gene_llama_embeddings, all_gene_names_llama,
                                                       num_macrogene=llama_macrogene_amount, normalize=False, seed=0)

    # Save macrogene maps for reproducibility
    with open(outdir / 'all-esm_to_macrogenes.pkl', 'wb') as f:
        pickle.dump(esm_macrogene_weights, f, protocol=4)
    with open(outdir / 'all-llama_to_macrogenes.pkl', 'wb') as f:
        pickle.dump(llama_macrogene_weights, f, protocol=4)

    # -----------------------------
    # STEP 5: centroid weights lists
    # -----------------------------
    esm_centroid_weights = []
    llama_centroid_weights = []
    for species in sorted_species_names:
        adata = species_to_adata_esm[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        species_gene_names = species_str.str.cat(gene_names, sep='_')
        for sgn in species_gene_names:
            esm_centroid_weights.append(torch.tensor(esm_macrogene_weights[sgn]))

    for species in sorted_species_names:
        adata = species_to_adata_llama[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        species_gene_names = species_str.str.cat(gene_names, sep='_')
        for sgn in species_gene_names:
            llama_centroid_weights.append(torch.tensor(llama_macrogene_weights[sgn]))

    esm_centroid_weights = torch.stack(esm_centroid_weights)     # [genes, esm_macros]
    llama_centroid_weights = torch.stack(llama_centroid_weights) # [genes, llama_macros]

    # Save processed adatas
    for k, v in species_to_adata_esm.items():
        v.write_h5ad(outdir / f'{k}_processed_esm.h5ad')
    for k, v in species_to_adata_llama.items():
        v.write_h5ad(outdir / f'{k}_processed_llama.h5ad')

    # -----------------------------
    # STEP 6: build macrogene adata (uses per-species counts)
    # -----------------------------
    macrogene_adata = {}
    num_species = len(sorted_species_names)
    llama_end_idx = 0
    esm_end_idx = 0

    for i in range(num_species):
        species = sorted_species_names[i]
        adata_llama = species_to_adata_llama[species]
        adata_esm = species_to_adata_esm[species]

        # ensure dense matrices for matmul
        Xl = adata_llama.X
        Xe = adata_esm.X
        Xl = Xl.toarray() if hasattr(Xl, "toarray") else Xl
        Xe = Xe.toarray() if hasattr(Xe, "toarray") else Xe

        adata_origin_llama = torch.tensor(Xl, dtype=torch.float64)
        adata_origin_esm = torch.tensor(Xe, dtype=torch.float64)

        llama_genes_per_species_target = adata_llama.shape[1]
        esm_genes_per_species_target = adata_esm.shape[1]

        if i == 0:
            llama_start_idx = 0
            llama_end_idx = llama_genes_per_species_target
        else:
            llama_start_idx = llama_end_idx
            llama_end_idx = llama_end_idx + llama_genes_per_species_target
        llama_weights = llama_centroid_weights[llama_start_idx:llama_end_idx, :]

        if i == 0:
            esm_start_idx = 0
            esm_end_idx = esm_genes_per_species_target
        else:
            esm_start_idx = esm_end_idx
            esm_end_idx = esm_end_idx + esm_genes_per_species_target
        esm_weights = esm_centroid_weights[esm_start_idx:esm_end_idx, :]

        llama_macrogene = adata_origin_llama @ llama_weights
        esm_macrogene = adata_origin_esm @ esm_weights

        macrogenes_input = torch.cat((esm_macrogene, llama_macrogene), dim=1)
        macrogene_adata[species] = ad.AnnData(X=macrogenes_input.numpy(), obs=adata_llama.obs.copy())
        macrogene_adata[species].write_h5ad(outdir / f'{species}_macrogene_adata.h5ad')

    # -----------------------------
    # STEP 7: dataset & dataloader
    # -----------------------------
    species_celltype_labels = {args.species_labels[i]: args.celltype_labels[i] for i in range(len(args.species_labels))}
    species_batch_labels = None if args.batch_labels == 'None' else args.batch_labels

    dataset = SingleCellDataset(species_to_adata=macrogene_adata,
                                species_celltype_labels=species_celltype_labels,
                                species_batch_labels=species_batch_labels)
    pin = torch.cuda.is_available()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin)
    gene_num = list(dataset.num_genes.values())[0]
    print("Gene num:", gene_num)

    # -----------------------------
    # STEP 8: models, losses, optimizers
    # -----------------------------
    encoder = Encoder(
        input_size=(args.num_esm_macrogene + args.num_llama_macrogene),
        hidden_size=args.macrogene_encoder_hidden_size,
        latent_size=args.latent_size,
        dropout_ratio=args.macrogene_encoder_dropout_ratio
    ).to(device)
    decoder = Decoder(
        gene_num, len(args.species_labels),
        hidden_size=args.macrogene_decoder_hidden_size,
        latent_size=args.latent_size,
        dropout_ratio=args.macrogene_decoder_dropout_ratio
    ).to(device)

    discriminator_species = Discriminator(
        latent_size=args.latent_size,
        hidden_size_adversary=args.hidden_size_adversary_species,
        species_num=len(args.species_labels)
    ).to(device)
    discriminator_celltype = Discriminator_celltype(
        latent_size=args.latent_size,
        hidden_size_adversary=args.hidden_size_adversary_celltype,
        celltype_num=dataset.cell_type_num
    ).to(device)

    reconstruction_loss = nn.MSELoss()
    adversarial_species_loss = nn.CrossEntropyLoss()
    adversarial_celltype_loss = nn.CrossEntropyLoss()
    discriminator_species_loss = nn.CrossEntropyLoss()
    discriminator_celltype_loss = nn.CrossEntropyLoss()

    import itertools
    optimizer_R = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr_generator)
    optimizer_D_species = torch.optim.Adam(discriminator_species.parameters(), lr=args.lr_discriminator_species)
    optimizer_A = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
    optimizer_D_celltype = torch.optim.Adam(discriminator_celltype.parameters(), lr=args.lr_discriminator_celltype)

    # -----------------------------
    # STEP 9: training loop with best tracking
    # -----------------------------
    print("Train epochs:", args.train_epochs)
    overall_celltype_ARI = 0.0  # last computed ARI

    for epoch in range(args.train_epochs):
        print(epoch)
        Reconstruction_loss_sum = 0.0
        encoder_loss_sum = 0.0
        adversary_species_loss_sum = 0.0
        adversary_celltype_loss_sum = 0.0
        overall_celltype_precision_sum = 0.0
        overall_species_precision_sum = 0.0
        overall_celltype_recall_sum = 0.0
        overall_species_recall_sum = 0.0
        overall_celltype_f1_sum = 0.0
        overall_species_f1_sum = 0.0
        overall_celltype_acc_sum = 0.0
        overall_species_acc_sum = 0.0

        encoder.train()
        decoder.train()
        discriminator_species.train()
        discriminator_celltype.train()

        for batch in dataloader:
            gene, species_onehot, species_idx, celltype_onehot, celltype_idx = batch
            gene = gene.to(device)
            species_idx = species_idx.to(device)
            species_onehot = species_onehot.to(device)
            celltype_onehot = celltype_onehot.to(device)
            celltype_idx = celltype_idx.to(device)

            # --- Pareto/MGDA ---
            optimizer_R.zero_grad()
            encoder.train(); discriminator_species.eval(); discriminator_celltype.eval()

            basal = encoder(gene)
            basal_species_onehot = torch.cat((basal, species_onehot), 1)
            recon = decoder(basal_species_onehot)

            recon_loss = reconstruction_loss(gene, recon)
            recon_loss_data = recon_loss.item()

            grad_encoder_recon = []
            recon_loss.backward()
            for p in encoder.parameters():
                grad_encoder_recon.append(p.grad.data.clone())
            encoder.zero_grad(); decoder.zero_grad(); discriminator_celltype.zero_grad(); discriminator_species.zero_grad()

            # species adv (negative)
            optimizer_A.zero_grad()
            encoder.train(); discriminator_species.eval(); discriminator_celltype.eval()
            fake_latent = encoder(gene)
            validity_fake_latent = discriminator_species(fake_latent)
            adv_loss = adversarial_species_loss(validity_fake_latent, species_idx)
            adv_loss_new = -adv_loss
            adv_species_loss_data = -1 * adv_loss.item()
            adv_loss_new.backward()
            grad_encoder_adv_species = []
            for p in encoder.parameters():
                if p.grad is not None:
                    grad_encoder_adv_species.append(p.grad.data.clone())
            encoder.zero_grad(); decoder.zero_grad(); discriminator_celltype.zero_grad(); discriminator_species.zero_grad()

            # celltype adv (positive), with optional masking
            optimizer_A.zero_grad()
            encoder.train(); discriminator_species.eval(); discriminator_celltype.eval()
            fake_latent = encoder(gene)
            # for annotation with a reference/query split (reference=0, query=1)
            if celltype_annotation_ref is not None:
                target_mask = (species_idx == 1)
                fake_latent = fake_latent[target_mask, :]
                celltype_idx = celltype_idx[target_mask]
            validity_fake_latent_celltype = discriminator_celltype(fake_latent)
            celltype_loss = adversarial_celltype_loss(validity_fake_latent_celltype, celltype_idx)
            adv_celltype_loss_data = celltype_loss.item()
            celltype_loss.backward()
            grad_encoder_adv_celltype = []
            for p in encoder.parameters():
                if p.grad is not None:
                    grad_encoder_adv_celltype.append(p.grad.data.clone())
            encoder.zero_grad(); decoder.zero_grad(); discriminator_celltype.zero_grad(); discriminator_species.zero_grad()

            # combine grads
            gn = gradient_normalizers(
                [grad_encoder_recon, grad_encoder_adv_species, grad_encoder_adv_celltype],
                [recon_loss_data, adv_species_loss_data, adv_celltype_loss_data],
                args.grad_normalized_type
            )
            grads = [grad_encoder_recon, grad_encoder_adv_species, grad_encoder_adv_celltype]
            for t in range(3):
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]
            sol, _ = MinNormSolver.find_min_norm_element(grads)
            alpha, beta, gamma = float(sol[0]), float(sol[1]), float(sol[2])
            gamma = max(0.4, gamma)
            denom = (alpha + beta) if (alpha + beta) != 0 else 1.0
            alpha_prop = alpha / denom
            beta_prop = beta / denom
            new_sum = 1.0 - gamma
            alpha = alpha_prop * new_sum
            beta = beta_prop * new_sum

            # Optimize reconstruction (R)
            optimizer_R.zero_grad()
            encoder.train(); discriminator_species.eval(); discriminator_celltype.eval()
            fake_latent = encoder(gene)
            fake_latent_speicies_onehot = torch.cat((fake_latent, species_onehot), 1)
            recon = decoder(fake_latent_speicies_onehot)
            recon_loss = reconstruction_loss(gene, recon)
            R_loss = alpha * recon_loss
            Reconstruction_loss_sum += R_loss.item()
            R_loss.backward()
            optimizer_R.step()

            # Optimize encoder (A)
            optimizer_A.zero_grad()
            encoder.train(); discriminator_species.eval(); discriminator_celltype.eval()
            fake_latent = encoder(gene)
            validity_fake_latent = discriminator_species(fake_latent)
            if celltype_annotation_ref is not None:
                target_mask = (species_idx == 1)
                fake_latent_masked = fake_latent[target_mask, :]
                validity_fake_latent_celltype = discriminator_celltype(fake_latent_masked)
            else:
                validity_fake_latent_celltype = discriminator_celltype(fake_latent)
            adv_loss = adversarial_species_loss(validity_fake_latent, species_idx)
            celltype_loss = adversarial_celltype_loss(validity_fake_latent_celltype, celltype_idx)
            A_loss = beta * (-adv_loss) + gamma * celltype_loss
            encoder_loss_sum += A_loss.item()
            A_loss.backward()
            optimizer_A.step()

            # Train species discriminator
            encoder.eval(); discriminator_celltype.eval(); discriminator_species.train()
            optimizer_D_species.zero_grad()
            fake_latent = encoder(gene)
            validity_fake_latent = discriminator_species(fake_latent)
            predicted_species_labels = np.argmax(validity_fake_latent.detach().cpu().numpy(), axis=1)
            sp_prec, sp_rec, sp_f1, sp_acc = compute_metrics(species_idx.cpu().numpy(), predicted_species_labels)
            overall_species_acc_sum += sp_acc
            overall_species_f1_sum += sp_f1
            overall_species_precision_sum += sp_prec
            overall_species_recall_sum += sp_rec
            D_loss = discriminator_species_loss(validity_fake_latent, species_idx)
            D_loss.backward()
            adversary_species_loss_sum += D_loss.item()
            optimizer_D_species.step()

            # Train celltype discriminator
            encoder.eval(); discriminator_species.eval(); discriminator_celltype.train()
            optimizer_D_celltype.zero_grad()
            fake_latent = encoder(gene)
            if celltype_annotation_ref is not None:
                target_mask = (species_idx == 1)
                fake_latent = fake_latent[target_mask, :]
            validity_fake_latent = discriminator_celltype(fake_latent)
            predicted_celltype_labels = np.argmax(validity_fake_latent.detach().cpu().numpy(), axis=1)
            ct_prec, ct_rec, ct_f1, ct_acc = compute_metrics(celltype_idx.cpu().numpy(), predicted_celltype_labels)
            overall_celltype_acc_sum += ct_acc
            overall_celltype_f1_sum += ct_f1
            overall_celltype_precision_sum += ct_prec
            overall_celltype_recall_sum += ct_rec
            D_celltype_loss = discriminator_celltype_loss(validity_fake_latent, celltype_idx)
            D_celltype_loss.backward()
            adversary_celltype_loss_sum += D_celltype_loss.item()
            optimizer_D_celltype.step()

        # --- epoch metrics ---
        n_batches = len(dataloader)
        Autoencoder_generator_loss = Reconstruction_loss_sum / n_batches
        Encoder_loss = encoder_loss_sum / n_batches
        Adversary_celltype_loss = adversary_celltype_loss_sum / n_batches
        Adversary_species_loss = adversary_species_loss_sum / n_batches
        Target_loss = Autoencoder_generator_loss + Encoder_loss
        overall_celltype_precision = overall_celltype_precision_sum / n_batches
        overall_species_precision = overall_species_precision_sum / n_batches
        overall_celltype_recall = overall_celltype_recall_sum / n_batches
        overall_species_recall = overall_species_recall_sum / n_batches
        overall_celltype_f1 = overall_celltype_f1_sum / n_batches
        overall_species_f1 = overall_species_f1_sum / n_batches
        overall_celltype_acc = overall_celltype_acc_sum / n_batches
        overall_species_acc = overall_species_acc_sum / n_batches

        # Evaluate ARI on embeddings at cadence
        if (epoch + 1) % eval_every == 0:
            overall_celltype_ARI_new = evaluation_model_embed(encoder, dataset=dataset)
            print(f'overall_celltype_ari_EMB (new): {overall_celltype_ARI_new:.6f}')
            encoder.eval(); decoder.eval(); discriminator_celltype.eval(); discriminator_species.eval()

            # Save snapshot this epoch
            #torch.save(encoder.state_dict(), outdir / f'esm_llama_encoder-{epoch}.pth')
            #torch.save(decoder.state_dict(), outdir / f'esm_llama_decoder-{epoch}.pth')
            #torch.save(discriminator_species.state_dict(), outdir / f'esm_llama_discriminator_species-{epoch}.pth')
            #torch.save(discriminator_celltype.state_dict(), outdir / f'esm_llama_discriminator_celltype-{epoch}.pth')

            # Update best checkpoint if improved
            if is_better(overall_celltype_ARI_new, best_metric, monitor_mode):
                best_metric = overall_celltype_ARI_new
                best_epoch = epoch
                save_checkpoint(best_path, encoder, decoder, epoch=epoch,
                                metrics={"overall_celltype_ARI_on_embeddings": float(best_metric)})
                with open(outdir / "best_meta.json", "w") as f:
                    json.dump({"epoch": int(best_epoch), "metric": "ARI_embed", "value": float(best_metric)}, f, indent=2)
                print(f"✅ New best at epoch {best_epoch}: ARI={best_metric:.6f} → {best_path}")

            overall_celltype_ARI = overall_celltype_ARI_new

        print(
            f"Epoch {epoch:03d} | AE: {Autoencoder_generator_loss:.4f} | "
            f"Dsp: {Adversary_species_loss:.4f} | Dct: {Adversary_celltype_loss:.4f} | "
            f"Target: {Target_loss:.4f} | ARI(embed): {overall_celltype_ARI:.4f}"
        )

        # Keep a compact epoch checkpoint (debug-friendly)
        save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", encoder, decoder, epoch=epoch, metrics={
            "AE_loss": Autoencoder_generator_loss,
            "Target_loss": Target_loss,
            "ARI_embed": float(overall_celltype_ARI)
        })

    # -----------------------------
    # End-to-end: reload best and generate outputs
    # -----------------------------
    if not best_path.exists():
        # Fallback: no ARI eval triggered / no improvement — use last epoch
        print("[E2E] best.pt not found. Using last-epoch weights for generation.")
        save_checkpoint(best_path, encoder, decoder, epoch=args.train_epochs - 1,
                        metrics={"overall_celltype_ARI_on_embeddings": float(overall_celltype_ARI)})
        best_epoch_used = args.train_epochs - 1
    else:
        ckpt = load_checkpoint(best_path, encoder, decoder, map_location=device)
        best_epoch_used = ckpt["epoch"]

    generate_outputs_from_models(encoder, decoder, dataset, outdir, best_epoch_used)
    merge_species_and_attach_latent(outdir, sorted_species_names)

if __name__ == "__main__":
    main()
