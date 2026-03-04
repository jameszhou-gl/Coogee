"""
Train a relation-aware graph encoder on per-code PrimeKG subgraphs,
produce a structural (hierarchical) embedding per medical code, and pair it with
a separately provided semantic embedding from ClinicalBERT.
"""

import os
import argparse
from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from torch_geometric.nn import RGCNConv
from transformers import AutoTokenizer, AutoModel

from scripts.utils import load_config
from scripts.token_semantic_desc_mapping import timegap_token_mapping_to_desc, special_tokens_mapping_to_desc

# -------------------------
# Utility: load global KG
# -------------------------
def load_kg(kg_csv_path: str):
    kg_df = pd.read_csv(kg_csv_path, low_memory=False)
    x = torch.as_tensor(kg_df["x_index"].values, dtype=torch.long)
    y = torch.as_tensor(kg_df["y_index"].values, dtype=torch.long)
    edge_index = torch.stack([x, y], dim=0)  # [2, E]

    rel_series = kg_df["display_relation"].astype(str).values
    rel_dict: Dict[str, int] = {}
    rel_ids = []
    for r in rel_series:
        if r not in rel_dict:
            rel_dict[r] = len(rel_dict)
        rel_ids.append(rel_dict[r])
    rel_index = torch.as_tensor(rel_ids, dtype=torch.long)  # [E]

    num_nodes_global = int(
        max(int(kg_df["x_index"].max()), int(kg_df["y_index"].max()))) + 1
    num_rels = len(rel_dict)
    print(
        f"[KG] edge_index: {edge_index.shape}, num_nodes_global={num_nodes_global}, num_rels={num_rels}")
    print(f"[KG] Number of nodes: {num_nodes_global}")
    print(f"[KG] Number of unique relations: {num_rels}")
    print(f"[KG] Edge index shape: {edge_index.shape}")
    print(f"[KG] Edge index range: [{edge_index.min()}, {edge_index.max()}]")
    print(f"[KG] Relation index shape: {rel_index.shape}")
    print(f"[KG] Relation index range: [{rel_index.min()}, {rel_index.max()}]")
    return edge_index, rel_index, num_nodes_global, num_rels


# -----------------------------------
# Dataset: per-code induced subgraph
# -----------------------------------
class HSCodeDataset(Dataset):
    """
    Each item is a PyG Data with:
    - x:                LongTensor [n_sub] GLOBAL node IDs for this subgraph
    - edge_index:       LongTensor [2, E_sub] LOCAL node IDs (relabel_nodes=True)
    - rel_index:        LongTensor [E_sub] relation IDs aligned to edge_index
    - anchor_idx:       LongTensor [n_anchor] LOCAL indices of anchor nodes for this code (may be empty)
    - med_code:         str
    - desc:             str
    """

    def __init__(self, vocab_codes_mappings_df: pd.DataFrame,
                 edge_index: torch.Tensor, rel_index: torch.Tensor,
                 relabel_nodes: bool = True):
        self.df = vocab_codes_mappings_df.reset_index(drop=True)
        self.edge_index_global = edge_index
        self.rel_index_global = rel_index.to(torch.long)
        self.relabel = relabel_nodes
        self.num_rels_global = int(rel_index.max().item())+1

        required_cols = ["med_code", "desc", "pkg_index_list"]
        for c in required_cols:
            assert c in self.df.columns, f"Missing column in vocab_codes_mappings: {c}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        nodes_l = list(row["pkg_index_list"])
        nodes_l.sort()
        nodes_tensor = torch.tensor(nodes_l, dtype=torch.long)
        
        # Get the subgraph
        sub_ei, sub_edge_type = subgraph(
            subset=nodes_tensor,
            edge_index=self.edge_index_global,
            edge_attr=self.rel_index_global,
            relabel_nodes=True
        )
        sub_edge_type = sub_edge_type.to(torch.long)
        
        # Validate subgraph structure
        n_nodes_sub = len(nodes_l)
        if sub_ei.numel() > 0:
            assert sub_ei.max() < n_nodes_sub, f"Edge index {sub_ei.max()} is out of bounds for subgraph with {n_nodes_sub} nodes"
            assert (sub_ei >= 0).all(), "Negative edge indices found in subgraph"
        
        anchors_local = torch.empty(0, dtype=torch.long)
        if "med_code_node_mapping" in self.df.columns and isinstance(row["med_code_node_mapping"], (list, tuple)):
            global2local = {g: j for j, g in enumerate(nodes_l)}
            anchors = [global2local[g]
                       for g in row["med_code_node_mapping"] if g in global2local]
            if anchors:
                anchors_local = torch.tensor(anchors, dtype=torch.long)

        data = Data(
            x=torch.tensor(nodes_l, dtype=torch.long),   # GLOBAL ids
            edge_index=sub_ei,                            # LOCAL ids
            edge_type=sub_edge_type,                           # relation ids
            anchor_idx=anchors_local,                     # LOCAL indices
            med_code=row["med_code"],
            desc=row["desc"]
        )
        return data


# -----------------------------------------
# Relation-aware encoder with global table
# -----------------------------------------
class RelGraphEncoder(nn.Module):
    def __init__(self, num_nodes_global: int, num_rels: int,
                 node_emb_dim: int = 256, hidden_dim: int = 256, out_dim: int = 256,
                 num_bases: int = 30, dropout: float = 0.1):
        super().__init__()
        self.global_emb = nn.Embedding(num_nodes_global, node_emb_dim)
        nn.init.normal_(self.global_emb.weight, std=0.02)
        self.conv1 = RGCNConv(node_emb_dim, hidden_dim,
                              num_relations=num_rels, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim,
                              num_relations=num_rels, num_bases=num_bases)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_global_ids: torch.Tensor, edge_index: torch.Tensor, rel_index: torch.Tensor):
        # Validation checks
        if rel_index.max() >= self.conv1.num_relations:
            raise ValueError(f"Relation index {rel_index.max()} is out of bounds for {self.conv1.num_relations} relations")
        if edge_index.max() >= len(x_global_ids):
            raise ValueError(f"Edge index {edge_index.max()} is out of bounds for {len(x_global_ids)} nodes")
            
        h = self.global_emb(x_global_ids)                    # [n_sub, d]
        h = F.relu(self.conv1(h, edge_index, rel_index))
        h = self.drop(h)
        h = self.conv2(h, edge_index, rel_index)             # [n_sub, out_dim]
        return h


# ------------------------------------------
# Link prediction loss (edge-wise BCE)
# ------------------------------------------
def edge_bce_loss(h: torch.Tensor,
                  edge_index: torch.Tensor,
                  num_neg: int = 1,
                  sampler_device: torch.device = None) -> torch.Tensor:
    """
    Compute binary cross-entropy over positive and sampled negative edges within this graph.
    - h:           [n_sub, d] node embeddings (local indices)
    - edge_index:  [2, E]     local edges
    - num_neg:     # negatives per positive
    Returns scalar loss.
    """
    # Positives
    src = edge_index[0]
    dst = edge_index[1]
    pos_score = (h[src] * h[dst]).sum(dim=-1)        # dot product
    pos_labels = torch.ones_like(pos_score)

    # Negatives: sample dst' for each src (within the same subgraph)
    n_nodes = h.size(0)
    E = src.numel()
    k = max(1, num_neg)
    # sample k negatives per positive
    neg_src = src.repeat_interleave(k)
    # random dst avoiding matching the true dst (simple re-sample loop)
    if sampler_device is None:
        sampler_device = h.device
    neg_dst = torch.randint(0, n_nodes, size=(E * k,), device=sampler_device)

    neg_score = (h[neg_src] * h[neg_dst]).sum(dim=-1)
    neg_labels = torch.zeros_like(neg_score)

    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    return F.binary_cross_entropy_with_logits(scores, labels)


# ------------------------------------------
# Pooling to per-code embedding
# ------------------------------------------
def pool_code_embedding(h: torch.Tensor, anchor_idx: torch.Tensor) -> torch.Tensor:
    """
    h:          [n_sub, d]
    anchor_idx: [n_anchor] LOCAL indices; may be empty
    """
    if anchor_idx is not None and anchor_idx.numel() > 0:
        return h[anchor_idx].mean(dim=0)
    return h.mean(dim=0)


# ------------------------------------------
# Model saving and loading
# ------------------------------------------
def save_gnn(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, knowledge_embd_dir: str):
    """Save GNN model checkpoint"""
    ckpt_dir = Path(knowledge_embd_dir)
    for f in ckpt_dir.glob("gnn_checkpoint_*.pt"): f.unlink()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, ckpt_dir / f"gnn_checkpoint_{epoch}.pt")
    print(f"[Checkpoint] Saved model to {knowledge_embd_dir}/gnn_checkpoint_{epoch}.pt, with loss {loss:.4f}")


def get_bert_embedding(text: str, model, tokenizer) -> np.ndarray:
    """Get ClinicalBERT embedding for a text string as numpy array"""
    # Get the device that the model is on
    device = next(model.parameters()).device

    # Tokenize and move to the same device as model
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    # Get [CLS] token embedding, move to CPU and convert to numpy
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

# --------------- Utilities ----------------
def get_latest_checkpoint(knowledge_embd_dir):
    cks = list(Path(knowledge_embd_dir).glob("gnn_checkpoint_*.pt"))
    if not cks: return None
    return str(max(cks, key=lambda p: int(p.stem.split("_")[-1])))

# ------------------------------------------
# Main Training
# ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--LOG_DIR", type=str, required=True,
                    help="Directory containing vocab_mapped_to_medtok.csv and output subdir")
    ap.add_argument("--kg_csv", type=str, default="artifacts/medtok_files/kg.csv")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--num_bases", type=int, default=30)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument('--train_again', action="store_true")
    args = ap.parse_args()
    knowledge_embd_dir = os.path.join(args.LOG_DIR, "knowledge_embd")
    os.makedirs(knowledge_embd_dir, exist_ok=True)
    cfg = load_config(os.path.join(args.LOG_DIR, "config.yaml"))
    if not (cfg["aux_embeddings"]["hierarchy"]["enabled"] or cfg["aux_embeddings"]["semantic"]["enabled"]):
        print("Both hierarchy and semantic embeddings are not enabled in config.yaml")
        exit()
    g_dim = cfg["aux_embeddings"]["hierarchy"]["dim"]
    bert_model_name = cfg["aux_embeddings"]["semantic"]["model"]

    # ! load knowledge_embd from a complete concept code, in artifacts/coogee_final/knowledge_embd
    pre_saved_knowledge_embd_dir = os.path.join("artifacts/coogee_final", "knowledge_embd")
    if not args.train_again and os.path.exists(os.path.join(pre_saved_knowledge_embd_dir, "hierarchy_embd.npz")):
        hierarchy_embeddings = np.load(os.path.join(pre_saved_knowledge_embd_dir, "hierarchy_embd.npz"))
        semantic_embeddings = np.load(os.path.join(pre_saved_knowledge_embd_dir, "semantic_embd.npz"))
        saved_concept_codes = list(hierarchy_embeddings.keys())
        print(f"Loaded hierarchy_embeddings and semantic_embeddings from: {os.path.join(pre_saved_knowledge_embd_dir)}")
        vocab = pl.read_csv(os.path.join(args.LOG_DIR, "knowledge_embd", "vocab_mapped_to_medtok.csv"))
        hierarchy_embedding_dict = {}
        semantic_embedding_dict = {}
        for token_id, concept_code, medtok_code in vocab.iter_rows():
            # The npz file stores arrays with keys as array names, so we need to use the string version
            if str(concept_code) in saved_concept_codes:
                hierarchy_embedding_dict[str(token_id)] = hierarchy_embeddings[str(concept_code)]
                semantic_embedding_dict[str(token_id)] = semantic_embeddings[str(concept_code)]
            else:
                print(f"Concept code {str(concept_code)} not found in hierarchy_embd.npz")
                hierarchy_embedding_dict[str(token_id)] = np.zeros(g_dim)
                semantic_embedding_dict[str(token_id)] = np.zeros(cfg["aux_embeddings"]["semantic"]["dim"])
        if cfg["aux_embeddings"]["semantic"]["enabled"]:
            np.savez(os.path.join(knowledge_embd_dir, "semantic_embd.npz"), **semantic_embedding_dict)
            print(f"Done. Saved semantic_embeddings to: {knowledge_embd_dir}")
        if cfg["aux_embeddings"]["hierarchy"]["enabled"]:
            np.savez(os.path.join(knowledge_embd_dir, "hierarchy_embd.npz"), **hierarchy_embedding_dict)
            print(f"Done. Saved hierarchy_embeddings to: {knowledge_embd_dir}")
        exit()
    wandb.init(project=cfg.get('wandb').get('project'), config=vars(args), name="construct_knowledge_embd")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {bert_model_name} model...")
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)
    bert_model.eval()
    
    vocab = pl.read_csv(os.path.join(args.LOG_DIR, "knowledge_embd", "vocab_mapped_to_medtok.csv"))
    all_codes_mappings = pl.read_parquet(os.path.join('artifacts/medtok_files', "all_codes_mappings.parquet"))
    valid_codes = vocab["medtok_code"].drop_nulls().to_list()
    print(f"{len(valid_codes)} in the vocabulary can be validly mapped to MedTok codes")
    vocab_codes_mappings = all_codes_mappings.filter(pl.col("med_code").is_in(valid_codes))
    print(f"{len(vocab_codes_mappings)} valid code mappings can be constructed")
    vocab_codes_mappings = vocab_codes_mappings.to_pandas()
    for col in ["pkg_index_list", "med_code_node_mapping"]:
        if col in vocab_codes_mappings.columns:
            vocab_codes_mappings[col] = vocab_codes_mappings[col].apply(lambda x: list(x) if isinstance(
                x, (list, tuple, np.ndarray)) else ([] if pd.isna(x) else [int(x)]))
    edge_index, rel_index, num_nodes_global, num_rels = load_kg(args.kg_csv)

    dataset = HSCodeDataset(vocab_codes_mappings, edge_index,
                            rel_index, relabel_nodes=True)

    def collate_fn(lst): return Batch.from_data_list(lst)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_fn)

    gnn = RelGraphEncoder(
        num_nodes_global=num_nodes_global,
        num_rels=num_rels,
        node_emb_dim=g_dim,
        hidden_dim=args.hidden,
        out_dim=g_dim,
        num_bases=args.num_bases,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(gnn.parameters(), lr=args.lr,
                            weight_decay=1e-2)

    start_epoch = 0
    best_loss = float('inf')
    
    ckpt_path = get_latest_checkpoint(knowledge_embd_dir)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        gnn.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt['loss']
        print(f"[ckpt] resumed from {ckpt_path}, epoch {start_epoch}, best_loss {best_loss}")
    print("[Train] Start")
    gnn.train()
    patience = 10
    for ep in range(start_epoch + 1, args.epochs + 1):
        running = 0.0
        n_items = 0
        for batch in tqdm(loader, desc=f"Training epoch {ep}/{args.epochs}"):
            batch = batch.to(device)
            
            h_all = gnn(batch.x, batch.edge_index,
                        batch.edge_type)  # [N_nodes_total, D]

            loss = torch.tensor(0.0, device=device)
            loss = edge_bce_loss(h_all, batch.edge_index,
                              num_neg=args.neg_per_pos, sampler_device=device)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            n_items += 1

        epoch_loss = running / max(1, n_items)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_gnn(gnn, opt, ep, epoch_loss, knowledge_embd_dir)
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                print(f"[Train] Patience reached, stopping training")
                break
        wandb.log({
            "epoch": ep,
            "loss": epoch_loss,
            "patience": patience
        })
    print("[Export] hierarchy embeddings per code...")
    gnn.eval()
    struct_embeds = []
    codes = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn), desc="Exporting structural embeddings"):
            batch = batch.to(device)
            h_all = gnn(batch.x, batch.edge_index, batch.edge_type)
            B = batch.num_graphs
            for g in range(B):
                start, end = batch.ptr[g].item(), batch.ptr[g + 1].item()
                h_g = h_all[start:end]
                med_code = batch.med_code[g]
                desc = batch.desc[g]
                anchors_local = torch.empty(0, dtype=torch.long, device=device)
                z_g = pool_code_embedding(h_g, anchors_local)
                struct_embeds.append(z_g.detach().cpu().numpy())
                codes.append(med_code)
    
    hierarchy_embedding_dict = {}
    semantic_embedding_dict = {}
    additional_tokens_descs = {**special_tokens_mapping_to_desc, **timegap_token_mapping_to_desc}
    for token_id, concept_code, medtok_code in tqdm(vocab.iter_rows(), total=len(vocab), desc="Building embeddings"):
        if medtok_code in codes:
            hierarchy_embedding_dict[str(concept_code)] = struct_embeds[codes.index(medtok_code)]
            code_desc = vocab_codes_mappings[vocab_codes_mappings["med_code"] == medtok_code]["desc"].values[0]
        else:
            hierarchy_embedding_dict[str(concept_code)] = np.zeros(g_dim)
            code_desc = additional_tokens_descs[concept_code]
        semantic_embedding_dict[str(concept_code)] = get_bert_embedding(
            code_desc, bert_model, bert_tokenizer)
    np.savez(os.path.join(knowledge_embd_dir, "hierarchy_embd.npz"), **hierarchy_embedding_dict)
    np.savez(os.path.join(knowledge_embd_dir, "semantic_embd.npz"), **semantic_embedding_dict)
    print(f"Done. Saved hierarchy_embeddings and semantic_embeddings to: {knowledge_embd_dir}")

if __name__ == "__main__":
    main()
