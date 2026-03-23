from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pickle
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.cluster import adjusted_rand_score
from torch import nn as torch_nn
from torch.utils.data import DataLoader

from dataset import SingleCellDataset, SingleCellDataset_for_raw_reconstruction, WeightedRandomSampler
from macrogene_initialize import load_gene_embeddings_adata, macrogene_initialization
from min_norm_solvers import MinNormSolver
from models import Decoder, Discriminator, Discriminator_celltype, Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Repro & small helpers
# -----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


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
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
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
    _adata.obs["celltype"] = y_true
    best_ari = 0.0
    for res in resolution:
        sc.tl.pca(_adata)
        _adata.obsm["X_emb"] = _adata.X
        sc.pp.neighbors(_adata, use_rep="X_emb")
        sc.tl.louvain(_adata, resolution=res, key_added="louvain")
        ari = adjusted_rand_score(y_true, _adata.obs["louvain"])
        if ari > best_ari:
            best_ari = ari
        del _adata.obs["louvain"]
    return float(best_ari)


# -----------------------------
# The gradient normalizer (kept)
# -----------------------------
def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == "l2":
        for t in range(len(grads)):
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == "loss":
        for t in range(len(grads)):
            gn[t] = losses[t]
    elif normalization_type == "loss+":
        for t in range(len(grads)):
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == "none":
        for t in range(len(grads)):
            gn[t] = 1.0
    else:
        print("ERROR: Invalid Normalization Type")
    return gn


# -----------------------------
# Common data loading
# -----------------------------
def load_species_data(args):
    species_to_adata = {
        args.species_labels[i]: sc.read_h5ad(args.h5ad_files[i])
        for i in range(len(args.species_labels))
    }
    species_to_gene_esm_embeddings_path = {
        args.species_labels[i]: args.gene_esm_embedding_path[i]
        for i in range(len(args.species_labels))
    }
    species_to_gene_llama_embeddings_path = {
        args.species_labels[i]: args.gene_llama_embedding_path[i]
        for i in range(len(args.species_labels))
    }
    return species_to_adata, species_to_gene_esm_embeddings_path, species_to_gene_llama_embeddings_path


# ============================================================
# Integration pipeline
# ============================================================
def evaluation_model_embed_integration(encoder: torch_nn.Module, dataset: SingleCellDataset) -> float:
    encoder.eval()
    gene_input = dataset.trans_profiles.to(device).float()
    with torch.no_grad():
        latent = encoder(gene_input)
    celltype_idx = dataset.celltype_id_embedding_all
    ari = compute_ARI_and_NMI_celltype(celltype_idx, latent.cpu().numpy())
    return float(ari)


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
        json.dump(
            {
                "epoch": int(epoch_used),
                "ARI_embeddings": float(ari_embed),
                "checkpoint": str(outdir / "best.pt"),
            },
            f,
            indent=2,
        )
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
        raise ValueError(
            f"latent rows ({latent.shape[0]}) != merged cells ({merged.n_obs}). "
            "Ensure the per-species ordering matches dataset construction."
        )

    # 5) Attach embedding
    merged.obsm["X_unify"] = latent

    # 6) Save
    merged_fp = outdir / "macrogene_merged_with_unify.h5ad"
    merged.write_h5ad(merged_fp)
    (outdir / "latent.npy").unlink()
    print(f"[E2E] Wrote merged macrogene AnnData with integrated emnbeddings to {merged_fp}")


def run_integration(args):
    # Seeds & paths
    set_seed(args.seed)
    outdir = Path(args.output_path)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best.pt"
    best_metric = None
    best_epoch = None
    monitor_mode = "max"  # maximize ARI
    eval_every = max(1, int(args.eval_every))

    # -----------------------------
    # STEP 1: load adata + gene embeddings
    # -----------------------------
    species_to_adata, species_to_gene_esm_embeddings_path, species_to_gene_llama_embeddings_path = load_species_data(args)

    species_to_adata_esm = copy.deepcopy(species_to_adata)
    species_to_adata_llama = copy.deepcopy(species_to_adata)

    species_to_gene_esm_embeddings = {}
    species_to_gene_llama_embeddings = {}

    for species, adata_item in species_to_adata_esm.items():
        adata_item, species_gene_esm_embeddings = load_gene_embeddings_adata(
            adata_item,
            species=[species],
            embedding_path=species_to_gene_esm_embeddings_path[species],
        )
        species_to_gene_esm_embeddings.update(species_gene_esm_embeddings)
        species_to_adata_esm[species] = adata_item
        print("Subsetting", species, "with ESM embeddings")

    for species, adata_item in species_to_adata_llama.items():
        adata_item, species_gene_llama_embeddings = load_gene_embeddings_adata(
            adata_item,
            species=[species],
            embedding_path=species_to_gene_llama_embeddings_path[species],
        )
        species_to_gene_llama_embeddings.update(species_gene_llama_embeddings)
        species_to_adata_llama[species] = adata_item
        print("Subsetting", species, "with LLaMA embeddings")

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
    ct = 0
    for species in sorted_species_names:
        adata_item = species_to_adata_esm[species]
        sc.pp.highly_variable_genes(adata_item, n_top_genes=high_variable_genes_esm, flavor="seurat_v3")
        hvg_index = adata_item.var["highly_variable"]
        species_to_adata_esm[species] = adata_item[:, hvg_index]
        species_to_gene_esm_embeddings[species] = species_to_gene_esm_embeddings[species][hvg_index]
        species_to_gene_idx_hvg_esm[species] = (ct, ct + species_to_gene_esm_embeddings[species].shape[0])
        ct += species_to_gene_esm_embeddings[species].shape[0]

    ct = 0
    for species in sorted_species_names:
        adata_item = species_to_adata_llama[species]
        sc.pp.highly_variable_genes(adata_item, flavor="seurat_v3", n_top_genes=high_variable_genes_llama)
        hvg_index = adata_item.var["highly_variable"]
        species_to_adata_llama[species] = adata_item[:, hvg_index]
        species_to_gene_llama_embeddings[species] = species_to_gene_llama_embeddings[species][hvg_index]
        species_to_gene_idx_hvg_llama[species] = (ct, ct + species_to_gene_llama_embeddings[species].shape[0])
        ct += species_to_gene_llama_embeddings[species].shape[0]

    # -----------------------------
    # STEP 3: concatenate names + embeddings
    # -----------------------------
    all_gene_names_esm = []
    for species in sorted_species_names:
        adata_item = species_to_adata_esm[species]
        species_str = pd.Series([species] * adata_item.var_names.shape[0])
        gene_names = pd.Series(adata_item.var_names)
        all_gene_names_esm += list(species_str.str.cat(gene_names, sep="_"))

    all_gene_names_llama = []
    for species in sorted_species_names:
        adata_item = species_to_adata_llama[species]
        species_str = pd.Series([species] * adata_item.var_names.shape[0])
        gene_names = pd.Series(adata_item.var_names)
        all_gene_names_llama += list(species_str.str.cat(gene_names, sep="_"))

    all_gene_esm_embeddings = torch.cat([species_to_gene_esm_embeddings[s] for s in sorted_species_names], dim=0)
    all_gene_llama_embeddings = torch.cat([species_to_gene_llama_embeddings[s] for s in sorted_species_names], dim=0)

    # -----------------------------
    # STEP 4: macrogene init
    # -----------------------------
    esm_macrogene_amount = args.num_esm_macrogene
    llama_macrogene_amount = args.num_llama_macrogene

    esm_macrogene_weights = macrogene_initialization(
        all_gene_esm_embeddings,
        all_gene_names_esm,
        num_macrogene=esm_macrogene_amount,
        normalize=False,
        seed=0,
    )
    llama_macrogene_weights = macrogene_initialization(
        all_gene_llama_embeddings,
        all_gene_names_llama,
        num_macrogene=llama_macrogene_amount,
        normalize=False,
        seed=0,
    )

    # Save macrogene maps for reproducibility
    with open(outdir / "all-esm_to_macrogenes.pkl", "wb") as f:
        pickle.dump(esm_macrogene_weights, f, protocol=4)
    with open(outdir / "all-llama_to_macrogenes.pkl", "wb") as f:
        pickle.dump(llama_macrogene_weights, f, protocol=4)

    # -----------------------------
    # STEP 5: centroid weights lists
    # -----------------------------
    esm_centroid_weights = []
    llama_centroid_weights = []
    for species in sorted_species_names:
        adata_item = species_to_adata_esm[species]
        species_str = pd.Series([species] * adata_item.var_names.shape[0])
        gene_names = pd.Series(adata_item.var_names)
        species_gene_names = species_str.str.cat(gene_names, sep="_")
        for sgn in species_gene_names:
            esm_centroid_weights.append(torch.tensor(esm_macrogene_weights[sgn]))

    for species in sorted_species_names:
        adata_item = species_to_adata_llama[species]
        species_str = pd.Series([species] * adata_item.var_names.shape[0])
        gene_names = pd.Series(adata_item.var_names)
        species_gene_names = species_str.str.cat(gene_names, sep="_")
        for sgn in species_gene_names:
            llama_centroid_weights.append(torch.tensor(llama_macrogene_weights[sgn]))

    esm_centroid_weights = torch.stack(esm_centroid_weights)
    llama_centroid_weights = torch.stack(llama_centroid_weights)

    # Save processed adatas
    for k, v in species_to_adata_esm.items():
        v.write_h5ad(outdir / f"{k}_processed_esm.h5ad")
    for k, v in species_to_adata_llama.items():
        v.write_h5ad(outdir / f"{k}_processed_llama.h5ad")

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
        macrogene_adata[species].write_h5ad(outdir / f"{species}_macrogene_adata.h5ad")

    # -----------------------------
    # STEP 7: dataset & dataloader
    # -----------------------------
    species_celltype_labels = {args.species_labels[i]: args.celltype_labels[i] for i in range(len(args.species_labels))}
    species_batch_labels = None if args.batch_labels == "None" else args.batch_labels

    dataset = SingleCellDataset(
        species_to_adata=macrogene_adata,
        species_celltype_labels=species_celltype_labels,
        species_batch_labels=species_batch_labels,
    )
    pin = torch.cuda.is_available()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin)
    gene_num = list(dataset.num_genes.values())[0]
    print("Total amount of macrogenes:", gene_num)

    # -----------------------------
    # STEP 8: models, losses, optimizers
    # -----------------------------
    encoder = Encoder(
        input_size=(args.num_esm_macrogene + args.num_llama_macrogene),
        hidden_size=args.macrogene_encoder_hidden_size,
        latent_size=args.latent_size,
        dropout_ratio=args.macrogene_encoder_dropout_ratio,
    ).to(device)
    decoder = Decoder(
        gene_num,
        len(args.species_labels),
        hidden_size=args.macrogene_decoder_hidden_size,
        latent_size=args.latent_size,
        dropout_ratio=args.macrogene_decoder_dropout_ratio,
    ).to(device)

    discriminator_species = Discriminator(
        latent_size=args.latent_size,
        hidden_size_adversary=args.hidden_size_adversary_species,
        species_num=len(args.species_labels),
    ).to(device)
    discriminator_celltype = Discriminator_celltype(
        latent_size=args.latent_size,
        hidden_size_adversary=args.hidden_size_adversary_celltype,
        celltype_num=dataset.cell_type_num,
    ).to(device)

    reconstruction_loss = nn.MSELoss()
    adversarial_species_loss = nn.CrossEntropyLoss()
    adversarial_celltype_loss = nn.CrossEntropyLoss()
    discriminator_species_loss = nn.CrossEntropyLoss()
    discriminator_celltype_loss = nn.CrossEntropyLoss()

    optimizer_R = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr_generator)
    optimizer_D_species = torch.optim.Adam(discriminator_species.parameters(), lr=args.lr_discriminator_species)
    optimizer_A = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
    optimizer_D_celltype = torch.optim.Adam(discriminator_celltype.parameters(), lr=args.lr_discriminator_celltype)

    # -----------------------------
    # STEP 9: training loop with best tracking
    # -----------------------------
    print("Train epochs:", args.train_epochs)
    overall_celltype_ARI = 0.0

    for epoch in range(args.train_epochs):
        Reconstruction_loss_sum = 0.0
        encoder_loss_sum = 0.0
        adversary_species_loss_sum = 0.0
        adversary_celltype_loss_sum = 0.0

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
                args.grad_normalized_type,
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
            _ = predicted_species_labels  # preserve original intermediate value usage

            D_loss = discriminator_species_loss(validity_fake_latent, species_idx)
            D_loss.backward()
            adversary_species_loss_sum += D_loss.item()
            optimizer_D_species.step()

            # Train celltype discriminator
            encoder.eval(); discriminator_species.eval(); discriminator_celltype.train()
            optimizer_D_celltype.zero_grad()
            fake_latent = encoder(gene)

            validity_fake_latent = discriminator_celltype(fake_latent)
            predicted_celltype_labels = np.argmax(validity_fake_latent.detach().cpu().numpy(), axis=1)
            _ = predicted_celltype_labels  # preserve original intermediate value usage

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

        # Evaluate ARI on embeddings at cadence
        if (epoch + 1) % eval_every == 0:
            overall_celltype_ARI_new = evaluation_model_embed_integration(encoder, dataset=dataset)
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
                save_checkpoint(
                    best_path,
                    encoder,
                    decoder,
                    epoch=epoch,
                    metrics={"overall_celltype_ARI_on_embeddings": float(best_metric)},
                )
                with open(outdir / "best_meta.json", "w") as f:
                    json.dump(
                        {"epoch": int(best_epoch), "metric": "ARI_embed", "value": float(best_metric)},
                        f,
                        indent=2,
                    )

            overall_celltype_ARI = overall_celltype_ARI_new

        #print(
        #    f"Epoch {epoch:03d} | AE: {Autoencoder_generator_loss:.4f} | "
        #    f"Dsp: {Adversary_species_loss:.4f} | Dct: {Adversary_celltype_loss:.4f} | "
        #    f"Target: {Target_loss:.4f} | ARI(embed): {overall_celltype_ARI:.4f}"
        #)

        # Keep a compact epoch checkpoint (debug-friendly)
        save_checkpoint(
            ckpt_dir / f"epoch_{epoch:04d}.pt",
            encoder,
            decoder,
            epoch=epoch,
            metrics={
                "AE_loss": Autoencoder_generator_loss,
                "Target_loss": Target_loss,
                "ARI_embed": float(overall_celltype_ARI),
            },
        )

    # -----------------------------
    # End-to-end: reload best and generate outputs
    # -----------------------------
    if not best_path.exists():
        # Fallback: no ARI eval triggered / no improvement — use last epoch
        print("[E2E] best.pt not found. Using last-epoch weights for generation.")
        save_checkpoint(
            best_path,
            encoder,
            decoder,
            epoch=args.train_epochs - 1,
            metrics={"overall_celltype_ARI_on_embeddings": float(overall_celltype_ARI)},
        )
        best_epoch_used = args.train_epochs - 1
    else:
        ckpt = load_checkpoint(best_path, encoder, decoder, map_location=device)
        best_epoch_used = ckpt["epoch"]

    generate_outputs_from_models(encoder, decoder, dataset, outdir, best_epoch_used)
    merge_species_and_attach_latent(outdir, sorted_species_names)

    for sp in sorted_species_names:
        fp = outdir / f"{sp}_macrogene_adata.h5ad"
        if fp.exists():
            fp.unlink()
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


# ============================================================
# Perturbation pipeline
# ============================================================
def custom_collate_fn(batch):
    raw_counts_list = [item[0] for item in batch]
    macrogenes_list = [item[1] for item in batch]
    species_onehot_list = [item[2] for item in batch]
    species_id_list = [item[3] for item in batch]
    celltype_onehot_list = [item[4] for item in batch]
    celltype_id_list = [item[5] for item in batch]
    batch_label_list = [item[6] for item in batch]

    raw_counts_batch = torch.stack(raw_counts_list)
    macrogenes_batch = torch.stack(macrogenes_list)
    species_onehot_batch = torch.from_numpy(np.stack(species_onehot_list))
    species_id_batch = torch.from_numpy(np.stack(species_id_list))
    celltype_onehot_batch = torch.from_numpy(np.stack(celltype_onehot_list))
    celltype_id_batch = torch.from_numpy(np.stack(celltype_id_list))

    if all(x is None for x in batch_label_list):
        collated_batch_labels = None
    else:
        collated_batch_labels = batch_label_list

    return (
        raw_counts_batch,
        macrogenes_batch,
        species_onehot_batch,
        species_id_batch,
        celltype_onehot_batch,
        celltype_id_batch,
        collated_batch_labels,
    )


# -----------------------------
# Inference model wrapper
# -----------------------------
class PerturbationInferenceModel(nn.Module):
    """
    Lightweight inference model that bundles encoder + decoder into one object.
    Discriminators are used only during training and are not saved in the final public model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_encoder: int,
        hidden_size_decoder: int,
        latent_size: int,
        encoder_dropout: float,
        decoder_dropout: float,
        output_size: int,
        species_num: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size_encoder,
            latent_size=latent_size,
            dropout_ratio=encoder_dropout,
        )
        self.decoder = Decoder(
            output_size,
            species_num,
            hidden_size=hidden_size_decoder,
            latent_size=latent_size,
            dropout_ratio=decoder_dropout,
        )

    def encode(self, macrogene_input: torch.Tensor) -> torch.Tensor:
        return self.encoder(macrogene_input)

    def decode(self, latent_input: torch.Tensor, species_onehot: torch.Tensor) -> torch.Tensor:
        decoder_input = torch.cat((latent_input, species_onehot), dim=1)
        return self.decoder(decoder_input)

    def forward(self, macrogene_input: torch.Tensor, species_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(macrogene_input)
        reconstruction = self.decode(latent, species_onehot)
        return latent, reconstruction


# -----------------------------
# Data preparation
# -----------------------------
def preprocess_species_data_perturbation(args):
    species_to_adata, species_to_gene_esm_embeddings_path, species_to_gene_llama_embeddings_path = load_species_data(args)

    species_to_adata_esm = copy.deepcopy(species_to_adata)
    species_to_adata_llama = copy.deepcopy(species_to_adata)
    species_to_gene_esm_embeddings = {}
    species_to_gene_llama_embeddings = {}

    for species, adata_item in species_to_adata_esm.items():
        adata_item, species_gene_esm_embeddings = load_gene_embeddings_adata(
            adata_item,
            species=[species],
            embedding_path=species_to_gene_esm_embeddings_path[species],
        )
        species_to_gene_esm_embeddings.update(species_gene_esm_embeddings)
        species_to_adata_esm[species] = adata_item
        print(f"[prep] subset {species} with genes that have ESM embeddings")

    for species, adata_item in species_to_adata_llama.items():
        adata_item, species_gene_llama_embeddings = load_gene_embeddings_adata(
            adata_item,
            species=[species],
            embedding_path=species_to_gene_llama_embeddings_path[species],
        )
        species_to_gene_llama_embeddings.update(species_gene_llama_embeddings)
        species_to_adata_llama[species] = adata_item
        print(f"[prep] subset {species} with genes that have LLaMA embeddings")

    sorted_species_names = sorted(list(species_to_adata.keys()))

    gene_amount_before_hvg_esm = max(v.shape[1] for v in species_to_adata_esm.values())
    gene_amount_before_hvg_llama = max(v.shape[1] for v in species_to_adata_llama.values())
    high_variable_genes_esm = min(args.highly_variable_genes, gene_amount_before_hvg_esm)
    high_variable_genes_llama = min(args.highly_variable_genes, gene_amount_before_hvg_llama)

    for species in sorted_species_names:
        adata_item = species_to_adata_esm[species].copy()
        sc.pp.highly_variable_genes(adata_item, n_top_genes=high_variable_genes_esm, flavor="seurat_v3")
        hvg_index = adata_item.var["highly_variable"].values
        species_to_adata_esm[species] = adata_item[:, hvg_index].copy()
        species_to_gene_esm_embeddings[species] = species_to_gene_esm_embeddings[species][hvg_index]

    for species in sorted_species_names:
        adata_item = species_to_adata_llama[species].copy()
        sc.pp.highly_variable_genes(adata_item, n_top_genes=high_variable_genes_llama, flavor="seurat_v3")
        hvg_index = adata_item.var["highly_variable"].values
        species_to_adata_llama[species] = adata_item[:, hvg_index].copy()
        species_to_gene_llama_embeddings[species] = species_to_gene_llama_embeddings[species][hvg_index]

    all_gene_names_esm = []
    all_gene_names_llama = []
    for species in sorted_species_names:
        esm_adata = species_to_adata_esm[species]
        all_gene_names_esm += list(pd.Series([species] * esm_adata.var_names.shape[0]).str.cat(pd.Series(esm_adata.var_names), sep="_"))

        llama_adata = species_to_adata_llama[species]
        all_gene_names_llama += list(pd.Series([species] * llama_adata.var_names.shape[0]).str.cat(pd.Series(llama_adata.var_names), sep="_"))

    all_gene_esm_embeddings = torch.cat(
        [species_to_gene_esm_embeddings[species] for species in sorted_species_names], dim=0
    )
    all_gene_llama_embeddings = torch.cat(
        [species_to_gene_llama_embeddings[species] for species in sorted_species_names], dim=0
    )

    esm_macrogene_weights = macrogene_initialization(
        all_gene_esm_embeddings,
        all_gene_names_esm,
        num_macrogene=args.num_esm_macrogene,
        normalize=False,
        seed=args.seed,
    )
    llama_macrogene_weights = macrogene_initialization(
        all_gene_llama_embeddings,
        all_gene_names_llama,
        num_macrogene=args.num_llama_macrogene,
        normalize=False,
        seed=args.seed,
    )

    esm_centroid_weights = {}
    llama_centroid_weights = {}
    for species in sorted_species_names:
        esm_adata = species_to_adata_esm[species]
        species_gene_names = pd.Series([species] * esm_adata.var_names.shape[0]).str.cat(
            pd.Series(esm_adata.var_names), sep="_"
        )
        esm_centroid_weights[species] = [torch.tensor(esm_macrogene_weights[sgn]) for sgn in species_gene_names]

        llama_adata = species_to_adata_llama[species]
        species_gene_names = pd.Series([species] * llama_adata.var_names.shape[0]).str.cat(
            pd.Series(llama_adata.var_names), sep="_"
        )
        llama_centroid_weights[species] = [torch.tensor(llama_macrogene_weights[sgn]) for sgn in species_gene_names]

    species_celltype_labels = {species: args.condition_key for species in sorted_species_names}
    species_batch_labels = None if args.batch_labels in [None, "None"] else args.batch_labels

    dataset = SingleCellDataset_for_raw_reconstruction(
        species_to_adata_esm=species_to_adata_esm,
        species_to_adata_llama=species_to_adata_llama,
        esm_centroid_weights=esm_centroid_weights,
        llama_centroid_weights=llama_centroid_weights,
        species_celltype_labels=species_celltype_labels,
        species_batch_labels=species_batch_labels,
    )

    return {
        "species_to_adata_esm": species_to_adata_esm,
        "species_to_adata_llama": species_to_adata_llama,
        "esm_centroid_weights": esm_centroid_weights,
        "llama_centroid_weights": llama_centroid_weights,
        "dataset": dataset,
        "sorted_species_names": sorted_species_names,
        "esm_macrogene_weights": esm_macrogene_weights,
        "llama_macrogene_weights": llama_macrogene_weights,
    }


def save_preprocessed_outputs(output_path, prep_obj):
    output_path = Path(output_path)
    with open(output_path / "all-gene_to_esm_macrogenes_weights.pkl", "wb") as f:
        pickle.dump(prep_obj["esm_macrogene_weights"], f, protocol=4)
    with open(output_path / "all-gene_to_llama_macrogenes_weights.pkl", "wb") as f:
        pickle.dump(prep_obj["llama_macrogene_weights"], f, protocol=4)

    for species, adata_item in prep_obj["species_to_adata_esm"].items():
        adata_item.write_h5ad(output_path / f"{species}_processed_esm.h5ad")
    for species, adata_item in prep_obj["species_to_adata_llama"].items():
        adata_item.write_h5ad(output_path / f"{species}_processed_llama.h5ad")

    dataset = prep_obj["dataset"]
    for species in prep_obj["sorted_species_names"]:
        macro = dataset.macrogenes_combined_per_species[species].to(dtype=torch.float32)
        macro_adata = ad.AnnData(X=macro.cpu().numpy(), obs=prep_obj["species_to_adata_esm"][species].obs.copy())
        #macro_adata.write_h5ad(output_path / f"{species}_macrogene_adata.h5ad")


# -----------------------------
# Model training and saving
# -----------------------------
def initialize_models(args, dataset):
    model = PerturbationInferenceModel(
        input_size=args.num_esm_macrogene + args.num_llama_macrogene,
        hidden_size_encoder=args.macrogene_encoder_hidden_size,
        hidden_size_decoder=args.macrogene_decoder_hidden_size,
        latent_size=args.latent_size,
        encoder_dropout=args.macrogene_encoder_dropout_ratio,
        decoder_dropout=args.macrogene_decoder_dropout_ratio,
        output_size=dataset.universal_raw_counts_stacked.shape[1],
        species_num=len(dataset.species_id_dict),
    ).to(device)

    discriminator_species = Discriminator(
        latent_size=args.latent_size,
        hidden_size_adversary=args.hidden_size_adversary_species,
        species_num=len(dataset.species_id_dict),
    ).to(device)

    discriminator_celltype = Discriminator_celltype(
        latent_size=args.latent_size,
        hidden_size_adversary=args.hidden_size_adversary_celltype,
        celltype_num=dataset.cell_type_num,
    ).to(device)

    return model, discriminator_species, discriminator_celltype


def build_inference_checkpoint(args, dataset, model) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": args.num_esm_macrogene + args.num_llama_macrogene,
            "hidden_size_encoder": args.macrogene_encoder_hidden_size,
            "hidden_size_decoder": args.macrogene_decoder_hidden_size,
            "latent_size": args.latent_size,
            "encoder_dropout": args.macrogene_encoder_dropout_ratio,
            "decoder_dropout": args.macrogene_decoder_dropout_ratio,
            "output_size": int(dataset.universal_raw_counts_stacked.shape[1]),
            "species_num": int(len(dataset.species_id_dict)),
        },
        "metadata": {
            "species_labels": sorted(dataset.species_id_dict.keys()),
            "species_id_dict": dataset.species_id_dict,
            "cell_type_id_dict": dataset.cell_type_id_dict,
            "num_genes_universe_per_species": dataset.num_genes_universe_per_species,
            "gene_names_universe": dataset.gene_names_universe,
            "source_species": args.source_species,
            "target_species": args.target_species,
            "condition_key": args.condition_key,
            "control_label": args.control_label,
            "perturb_label": args.perturb_label,
            "prediction_outputs": ["basal", "macrogene"],
        },
    }


def save_final_inference_model(output_path, args, dataset, model):
    checkpoint = build_inference_checkpoint(args, dataset, model)
    save_path = Path(output_path) / "final_inference_model.pt"
    torch.save(checkpoint, save_path)
    return save_path


def train_model_perturbation(args, dataset, output_path):
    sampler = WeightedRandomSampler(weights=dataset.sample_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=True,
    )

    model, discriminator_species, discriminator_celltype = initialize_models(args, dataset)

    reconstruction_loss = torch.nn.MSELoss()
    adversarial_species_loss = torch.nn.CrossEntropyLoss()
    adversarial_celltype_loss = torch.nn.CrossEntropyLoss()
    discriminator_species_loss = torch.nn.CrossEntropyLoss()
    discriminator_celltype_loss = torch.nn.CrossEntropyLoss()

    optimizer_recon = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=args.lr_generator,
    )
    optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=args.lr_encoder)
    optimizer_D_species = torch.optim.Adam(discriminator_species.parameters(), lr=args.lr_discriminator_species)
    optimizer_D_celltype = torch.optim.Adam(discriminator_celltype.parameters(), lr=args.lr_discriminator_celltype)





    history = []
    for epoch in range(args.train_epochs):
        model.train()
        discriminator_species.train()
        discriminator_celltype.train()

        recon_running = 0.0
        encoder_running = 0.0
        species_disc_running = 0.0
        celltype_disc_running = 0.0

        for batch in dataloader:
            universal_raw_counts_batch, gene, species_onehot, species_idx, _, celltype_idx, _ = batch

            gene = gene.to(device=device, dtype=torch.float32)
            universal_raw_counts_batch = universal_raw_counts_batch.to(device=device, dtype=torch.float32)
            species_onehot = species_onehot.to(device=device, dtype=torch.float32)
            species_idx = species_idx.to(device=device, dtype=torch.long)
            celltype_idx = celltype_idx.to(device=device, dtype=torch.long)

            if species_idx.ndim > 1:
                species_idx = species_idx.squeeze(-1)
            if celltype_idx.ndim > 1:
                celltype_idx = celltype_idx.squeeze(-1)

            # step 1: reconstruction objective for encoder + decoder
            optimizer_recon.zero_grad()
            fake_latent, recon = model(gene, species_onehot)
            loss_recon = 1 * reconstruction_loss(universal_raw_counts_batch, recon)
            loss_recon.backward()
            optimizer_recon.step()
            recon_running += loss_recon.item()

            # step 2: adversarial / condition objective on encoder
            optimizer_encoder.zero_grad()
            fake_latent = model.encode(gene)
            validity_species = discriminator_species(fake_latent)
            validity_celltype = discriminator_celltype(fake_latent)
            loss_adv_species = adversarial_species_loss(validity_species, species_idx)
            loss_adv_celltype = adversarial_celltype_loss(validity_celltype, celltype_idx)
            loss_encoder = (-0.01 * loss_adv_species) + (1 * loss_adv_celltype)
            loss_encoder.backward()
            optimizer_encoder.step()
            encoder_running += loss_encoder.item()

            # step 3: species discriminator
            optimizer_D_species.zero_grad()
            with torch.no_grad():
                fake_latent_detached = model.encode(gene)
            validity_species = discriminator_species(fake_latent_detached)
            loss_species_disc = discriminator_species_loss(validity_species, species_idx)
            loss_species_disc.backward()
            optimizer_D_species.step()
            species_disc_running += loss_species_disc.item()

            # step 4: condition discriminator
            optimizer_D_celltype.zero_grad()
            with torch.no_grad():
                fake_latent_detached = model.encode(gene)
            validity_celltype = discriminator_celltype(fake_latent_detached)
            loss_celltype_disc = discriminator_celltype_loss(validity_celltype, celltype_idx)
            loss_celltype_disc.backward()
            optimizer_D_celltype.step()
            celltype_disc_running += loss_celltype_disc.item()

        epoch_record = {
            "epoch": epoch + 1,
            "reconstruction_loss": recon_running / len(dataloader),
            "encoder_loss": encoder_running / len(dataloader),
            "species_discriminator_loss": species_disc_running / len(dataloader),
            "condition_discriminator_loss": celltype_disc_running / len(dataloader),
        }
        history.append(epoch_record)

    #pd.DataFrame(history).to_csv(Path(output_path) / "training_history.csv", index=False)
    final_model_path = save_final_inference_model(output_path, args, dataset, model.eval())
    return model.eval(), final_model_path


def load_saved_inference_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = PerturbationInferenceModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# -----------------------------
# Prediction
# -----------------------------
def compute_basal_embeddings(dataset, model):
    model.eval()
    batch_size = 512
    all_basal = []
    for i in range(0, dataset.macrogenes_combined.shape[0], batch_size):
        batch = dataset.macrogenes_combined[i : i + batch_size].to(dtype=torch.float32, device=device)
        with torch.no_grad():
            basal_embedding = model.encode(batch)
        all_basal.append(basal_embedding.cpu())

    all_basal = torch.cat(all_basal, dim=0).numpy()
    id_to_condition = {v: k for k, v in dataset.cell_type_id_dict.items()}
    id_to_species = {v: k for k, v in dataset.species_id_dict.items()}

    condition_labels = [id_to_condition[i] for i in dataset.global_celltype_id_embedding]
    species_labels = [id_to_species[i] for i in dataset.global_species_id_embedding]

    adata_basal = ad.AnnData(all_basal)
    adata_basal.obs["species"] = species_labels
    adata_basal.obs["condition"] = condition_labels
    return adata_basal


def get_species_gene_slice(dataset, species_name):
    sorted_species = sorted(dataset.num_genes_universe_per_species.keys())
    start = 0
    for sp_name in sorted_species:
        n_genes = dataset.num_genes_universe_per_species[sp_name]
        if sp_name == species_name:
            return start, start + n_genes
        start += n_genes
    raise ValueError(f"Species {species_name} not found in dataset.num_genes_universe_per_species")


def build_species_onehot(dataset, target_species, n_cells):
    sorted_species = sorted(dataset.species_id_dict.keys())
    species_idx = sorted_species.index(target_species)
    onehot = np.zeros((n_cells, len(sorted_species)), dtype=np.float32)
    onehot[:, species_idx] = 1.0
    return torch.tensor(onehot, dtype=torch.float32, device=device)


def decode_to_species_space(latent_input, model, dataset, target_species):
    species_onehot = build_species_onehot(dataset, target_species, latent_input.shape[0])
    latent_input = latent_input.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_all = model.decode(latent_input, species_onehot)

    start, end = get_species_gene_slice(dataset, target_species)
    pred_target = pred_all[:, start:end].detach().cpu().numpy()

    gene_names = dataset.gene_names_universe[target_species]
    gene_names = [g.replace(f"{target_species}_", "") for g in gene_names]
    adata_pred = ad.AnnData(pred_target)
    adata_pred.var_names = gene_names
    return adata_pred


def subset_species_condition(adata_obj, species_name, condition_name):
    mask = (adata_obj.obs["species"] == species_name) & (adata_obj.obs["condition"] == condition_name)
    return adata_obj[mask].copy()


def add_prediction_metadata(pred_adata, args, prediction_mode):
    pred_adata.obs[args.condition_key] = args.perturb_label
    pred_adata.obs["prediction_mode"] = prediction_mode
    pred_adata.obs["source_species"] = args.source_species
    pred_adata.obs["target_species"] = args.target_species
    pred_adata.obs["source_control_label"] = args.control_label
    pred_adata.obs["source_perturb_label"] = args.perturb_label
    return pred_adata


def predict_target_perturbation_basal(dataset, model, args):
    adata_basal = compute_basal_embeddings(dataset, model)

    source_control = subset_species_condition(adata_basal, args.source_species, args.control_label)
    source_perturb = subset_species_condition(adata_basal, args.source_species, args.perturb_label)
    target_control = subset_species_condition(adata_basal, args.target_species, args.control_label)

    if source_control.n_obs == 0 or source_perturb.n_obs == 0 or target_control.n_obs == 0:
        raise ValueError("Could not find the required source/target control/perturb cells in the basal embeddings.")

    source_control_mean = np.mean(to_numpy(source_control.X), axis=0).reshape(1, -1)
    target_control_mean = np.mean(to_numpy(target_control.X), axis=0).reshape(1, -1)
    source_perturb_basal = to_numpy(source_perturb.X)

    # basal-based prediction uses the same linear-transformation strategy as the
    # reference prediction script, rather than the target-control + delta rule
    # used for macrogene-based prediction.
    lr = LinearRegression(fit_intercept=True)
    lr.fit(source_control_mean, target_control_mean)
    pred_target_basal = lr.predict(source_perturb_basal)
    pred_target_basal = torch.tensor(pred_target_basal, dtype=torch.float32)

    pred_adata = decode_to_species_space(pred_target_basal, model, dataset, args.target_species)
    return add_prediction_metadata(pred_adata, args, prediction_mode="basal")


def predict_target_perturbation_macrogene(dataset, model, args, species_to_adata_esm):
    source_macro = dataset.macrogenes_combined_per_species[args.source_species].to(dtype=torch.float32)
    source_macro_adata = ad.AnnData(X=source_macro.cpu().numpy(), obs=species_to_adata_esm[args.source_species].obs.copy())

    target_macro = dataset.macrogenes_combined_per_species[args.target_species].to(dtype=torch.float32)
    target_macro_adata = ad.AnnData(X=target_macro.cpu().numpy(), obs=species_to_adata_esm[args.target_species].obs.copy())

    source_macro_control = source_macro_adata[source_macro_adata.obs[args.condition_key] == args.control_label].copy()
    source_macro_perturb = source_macro_adata[source_macro_adata.obs[args.condition_key] == args.perturb_label].copy()
    target_macro_control = target_macro_adata[target_macro_adata.obs[args.condition_key] == args.control_label].copy()

    if source_macro_control.n_obs == 0 or source_macro_perturb.n_obs == 0 or target_macro_control.n_obs == 0:
        raise ValueError("Could not find the required source/target control/perturb cells in the macrogene inputs.")

    delta = np.mean(to_numpy(source_macro_perturb.X), axis=0) - np.mean(to_numpy(source_macro_control.X), axis=0)
    pred_target_macro = to_numpy(target_macro_control.X) + delta
    pred_target_macro = torch.tensor(pred_target_macro, dtype=torch.float32)

    with torch.no_grad():
        pred_target_basal = model.encode(pred_target_macro.to(device))
    pred_adata = decode_to_species_space(pred_target_basal, model, dataset, args.target_species)
    return add_prediction_metadata(pred_adata, args, prediction_mode="macrogene")


def run_perturbation(args):
    set_seed(args.seed)
    ensure_dir(args.output_path)

    # step 1: preprocess input data and build the training dataset
    prep_obj = preprocess_species_data_perturbation(args)
    save_preprocessed_outputs(args.output_path, prep_obj)
    dataset = prep_obj["dataset"]

    # step 2: train the model end-to-end and save a single final inference model
    model, final_model_path = train_model_perturbation(args, dataset, args.output_path)

    # optional sanity check: reload the one-file model before prediction
    model, _ = load_saved_inference_model(final_model_path)

    # step 3: output both prediction types for the target species perturbation
    basal_pred = predict_target_perturbation_basal(dataset, model, args)
    basal_path = Path(args.output_path) / f"{args.target_species}_predicted_{args.perturb_label}_basal.h5ad"
    basal_pred.write_h5ad(basal_path)

    macrogene_pred = predict_target_perturbation_macrogene(
        dataset,
        model,
        args,
        prep_obj["species_to_adata_esm"],
    )
    macrogene_path = Path(args.output_path) / f"{args.target_species}_predicted_{args.perturb_label}_macrogene.h5ad"
    macrogene_pred.write_h5ad(macrogene_path)

    print("[done] Training and prediction finished.")
    print(f"[done] Single final model saved to: {final_model_path}")
    print(f"[done] Basal-based prediction saved to: {basal_path}")
    print(f"[done] Macrogene-based prediction saved to: {macrogene_path}")


# ============================================================
# Argument parsing and routing
# ============================================================
TASK_DEFAULTS = {
    "integration": {
        "grad_normalized_type": "l2",
        "eval_every": 5,
        "batch_size": 256,
        "hidden_size_adversary_species": 256,
        "hidden_size_adversary_celltype": 256,
        "latent_size": 256,
        "lr_discriminator_celltype": 0.00018364960390400008,
        "lr_discriminator_species": 0.00015139547698955852,
        "lr_encoder": 0.0008269200693451001,
        "lr_generator": 0.00010667566651855372,
        "macrogene_decoder_dropout_ratio": 0.3,
        "macrogene_decoder_hidden_size": 256,
        "macrogene_encoder_dropout_ratio": 0.2,
        "macrogene_encoder_hidden_size": 256,
        "train_epochs": 500,
    },
    "perturbation": {
        "batch_size": 128,
        "hidden_size_adversary_species": 256,
        "hidden_size_adversary_celltype": 256,
        "latent_size": 256,
        "lr_discriminator_celltype": 1e-3,
        "lr_discriminator_species": 1e-3,
        "lr_encoder": 1e-3,
        "lr_generator": 1e-3,
        "macrogene_decoder_dropout_ratio": 0.3,
        "macrogene_decoder_hidden_size": 256,
        "macrogene_encoder_dropout_ratio": 0.2,
        "macrogene_encoder_hidden_size": 256,
        "train_epochs": 1000,
    },
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified Unify pipeline. Choose mode = integration or perturbation."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["integration", "perturbation"],
        help="Which workflow to run.",
    )

    parser.add_argument("--h5ad_files", nargs="+", required=True, help="List of h5ad files, one per species")
    parser.add_argument("--species_labels", nargs="+", required=True, help="Species names in the same order as --h5ad_files")
    parser.add_argument("--gene_esm_embedding_path", nargs="+", required=True, help="List of gene ESM embedding files")
    parser.add_argument("--gene_llama_embedding_path", nargs="+", required=True, help="List of gene LLaMA embedding files")
    parser.add_argument("--output_path", type=str, required=True)

    # integration-specific
    parser.add_argument("--celltype_labels", nargs="+", default=None, help="Integration only: obs column names for cell-type labels")
    parser.add_argument("--grad_normalized_type", type=str, default=None, help="Integration only. If omitted, uses the integration default.")
    parser.add_argument("--eval_every", type=int, default=None, help="Integration only: evaluate ARI every N epochs. If omitted, uses the integration default.")

    # perturbation-specific
    parser.add_argument("--source_species", type=str, default=None, help="Perturbation only: species used as the perturbation reference")
    parser.add_argument("--target_species", type=str, default=None, help="Perturbation only: species to predict perturbation for")
    parser.add_argument("--condition_key", type=str, default=None, help="Perturbation only: obs column storing control/perturb labels")
    parser.add_argument("--control_label", type=str, default=None, help="Perturbation only: control condition label")
    parser.add_argument("--perturb_label", type=str, default=None, help="Perturbation only: perturbation condition label")

    # shared model / preprocessing args
    parser.add_argument("--num_esm_macrogene", type=int, default=2000)
    parser.add_argument("--num_llama_macrogene", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--highly_variable_genes", type=int, default=8000)
    parser.add_argument("--batch_labels", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--hidden_size_adversary_species", type=int, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--hidden_size_adversary_celltype", type=int, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--latent_size", type=int, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--lr_discriminator_celltype", type=float, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--lr_discriminator_species", type=float, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--lr_encoder", type=float, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--lr_generator", type=float, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--macrogene_decoder_dropout_ratio", type=float, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--macrogene_decoder_hidden_size", type=int, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--macrogene_encoder_dropout_ratio", type=float, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--macrogene_encoder_hidden_size", type=int, default=None, help="If omitted, uses the task-specific default.")
    parser.add_argument("--train_epochs", type=int, default=None, help="If omitted, uses the task-specific default.")
    return parser


def apply_task_defaults(args):
    task_defaults = TASK_DEFAULTS[args.task]
    for key, value in task_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def validate_args(args):
    if len(args.h5ad_files) != len(args.species_labels):
        raise ValueError("--h5ad_files and --species_labels must have the same length")
    if len(args.gene_esm_embedding_path) != len(args.species_labels):
        raise ValueError("--gene_esm_embedding_path and --species_labels must have the same length")
    if len(args.gene_llama_embedding_path) != len(args.species_labels):
        raise ValueError("--gene_llama_embedding_path and --species_labels must have the same length")

    if args.task == "integration":
        if args.celltype_labels is None:
            raise ValueError("Integration mode requires --celltype_labels")
        if len(args.celltype_labels) != len(args.species_labels):
            raise ValueError("--celltype_labels and --species_labels must have the same length in integration mode")

    if args.task == "perturbation":
        required = [
            "source_species",
            "target_species",
            "condition_key",
            "control_label",
            "perturb_label",
        ]
        missing = [name for name in required if getattr(args, name) in [None, ""]]
        if missing:
            raise ValueError(f"Perturbation mode requires: {', '.join(missing)}")
        if args.source_species not in args.species_labels or args.target_species not in args.species_labels:
            raise ValueError("--source_species and --target_species must be present in --species_labels")
        if args.source_species == args.target_species:
            raise ValueError("--source_species and --target_species must be different")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    args.task = args.mode
    args = apply_task_defaults(args)
    validate_args(args)

    if args.task == "integration":
        run_integration(args)
    elif args.task == "perturbation":
        run_perturbation(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
