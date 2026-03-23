import argparse
import copy
import pickle
import random
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SingleCellDataset_for_raw_reconstruction, WeightedRandomSampler
from macrogene_initialize import macrogene_initialization, load_gene_embeddings_adata
from models import Decoder, Discriminator, Discriminator_celltype, Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utilities
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


def preprocess_species_data(args):
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

    species_celltype_labels = {
        species: args.condition_key for species in sorted_species_names
    }
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
    with open(output_path / "all-esm_to_macrogenes.pkl", "wb") as f:
        pickle.dump(prep_obj["esm_macrogene_weights"], f, protocol=4)
    with open(output_path / "all-llama_to_macrogenes.pkl", "wb") as f:
        pickle.dump(prep_obj["llama_macrogene_weights"], f, protocol=4)
    with open(output_path / "esm_centroid_weights.pkl", "wb") as f:
        pickle.dump(prep_obj["esm_centroid_weights"], f, protocol=4)
    with open(output_path / "llama_centroid_weights.pkl", "wb") as f:
        pickle.dump(prep_obj["llama_centroid_weights"], f, protocol=4)

    for species, adata_item in prep_obj["species_to_adata_esm"].items():
        adata_item.write_h5ad(output_path / f"{species}_processed_esm.h5ad")
    for species, adata_item in prep_obj["species_to_adata_llama"].items():
        adata_item.write_h5ad(output_path / f"{species}_processed_llama.h5ad")

    dataset = prep_obj["dataset"]
    for species in prep_obj["sorted_species_names"]:
        macro = dataset.macrogenes_combined_per_species[species].to(dtype=torch.float32)
        macro_adata = ad.AnnData(X=macro.cpu().numpy(), obs=prep_obj["species_to_adata_esm"][species].obs.copy())
        macro_adata.write_h5ad(output_path / f"{species}_macrogene_adata.h5ad")


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


def train_model(args, dataset, output_path):
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

    alpha = args.alpha_recon
    beta = args.beta_species
    gamma = args.gamma_celltype

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
            loss_recon = alpha * reconstruction_loss(universal_raw_counts_batch, recon)
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
            loss_encoder = (-beta * loss_adv_species) + (gamma * loss_adv_celltype)
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
        print(
            f"Epoch {epoch + 1}/{args.train_epochs} | "
            f"Recon: {epoch_record['reconstruction_loss']:.4f} | "
            f"Enc: {epoch_record['encoder_loss']:.4f} | "
            f"D_species: {epoch_record['species_discriminator_loss']:.4f} | "
            f"D_condition: {epoch_record['condition_discriminator_loss']:.4f}"
        )

    pd.DataFrame(history).to_csv(Path(output_path) / "training_history.csv", index=False)
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

    delta = np.mean(source_perturb.X, axis=0) - np.mean(source_control.X, axis=0)
    pred_target_basal = target_control.X + delta
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end perturbation pipeline: preprocess, train, save one final inference model, and output basal/macrogene predictions"
    )
    parser.add_argument("--h5ad_files", nargs="+", required=True, help="List of h5ad files, one per species")
    parser.add_argument("--species_labels", nargs="+", required=True, help="Species names in the same order as --h5ad_files")
    parser.add_argument("--gene_esm_embedding_path", nargs="+", required=True, help="List of gene ESM embedding files")
    parser.add_argument("--gene_llama_embedding_path", nargs="+", required=True, help="List of gene LLaMA embedding files")

    parser.add_argument("--source_species", type=str, required=True, help="Species used as the perturbation reference")
    parser.add_argument("--target_species", type=str, required=True, help="Species to predict perturbation for")
    parser.add_argument("--condition_key", type=str, required=True, help="obs column storing control/perturb condition labels")
    parser.add_argument("--control_label", type=str, required=True, help="Control condition label")
    parser.add_argument("--perturb_label", type=str, required=True, help="Perturbation condition label")

    parser.add_argument("--num_esm_macrogene", type=int, default=2000)
    parser.add_argument("--num_llama_macrogene", type=int, default=2000)
    parser.add_argument("--highly_variable_genes", type=int, default=8000)
    parser.add_argument("--batch_labels", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size_adversary_species", type=int, default=256)
    parser.add_argument("--hidden_size_adversary_celltype", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=256)
    parser.add_argument("--lr_discriminator_celltype", type=float, default=1e-3)
    parser.add_argument("--lr_discriminator_species", type=float, default=1e-3)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_generator", type=float, default=1e-3)
    parser.add_argument("--macrogene_decoder_dropout_ratio", type=float, default=0.3)
    parser.add_argument("--macrogene_decoder_hidden_size", type=int, default=256)
    parser.add_argument("--macrogene_encoder_dropout_ratio", type=float, default=0.2)
    parser.add_argument("--macrogene_encoder_hidden_size", type=int, default=256)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--alpha_recon", type=float, default=1.0)
    parser.add_argument("--beta_species", type=float, default=0.01)
    parser.add_argument("--gamma_celltype", type=float, default=1.0)

    args = parser.parse_args()

    if len(args.h5ad_files) != len(args.species_labels):
        raise ValueError("--h5ad_files and --species_labels must have the same length")
    if len(args.gene_esm_embedding_path) != len(args.species_labels):
        raise ValueError("--gene_esm_embedding_path and --species_labels must have the same length")
    if len(args.gene_llama_embedding_path) != len(args.species_labels):
        raise ValueError("--gene_llama_embedding_path and --species_labels must have the same length")
    if args.source_species not in args.species_labels or args.target_species not in args.species_labels:
        raise ValueError("--source_species and --target_species must be present in --species_labels")
    if args.source_species == args.target_species:
        raise ValueError("--source_species and --target_species must be different")

    set_seed(args.seed)
    ensure_dir(args.output_path)

    # step 1: preprocess input data and build the training dataset
    prep_obj = preprocess_species_data(args)
    save_preprocessed_outputs(args.output_path, prep_obj)
    dataset = prep_obj["dataset"]

    # step 2: train the model end-to-end and save a single final inference model
    model, final_model_path = train_model(args, dataset, args.output_path)

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


if __name__ == "__main__":
    main()
