'''
    code by Guanglin Zhou (jameszhou.ustc@gmail.com)
    Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
    17 Sep 2025: Add KV-cache for efficient inference;
'''
import math, json, os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_auxiliary_embeddings(embd_path: str, vocab_size: int, embd_dim: int, embd_type: str) -> torch.nn.Parameter:
    """Load auxiliary embeddings (hierarchy, or semantic) from npz file.
    
    Args:
        embd_path: Path to the npz file containing embeddings
        vocab_size: Size of the vocabulary
        embd_dim: Dimension of the embeddings
        embd_type: Type of embeddings (for logging purposes)
        
    Returns:
        torch.nn.Parameter: Frozen embedding matrix of shape (vocab_size, embd_dim)
    """
    embeddings = np.zeros((vocab_size, embd_dim))
    
    with np.load(embd_path) as data:
        for token_id_str in data.files:
            token_id = int(token_id_str)
            embeddings[token_id] = data[token_id_str]
    
    print(f"Loaded {embd_type} embeddings for {len(data.files)} tokens")
    
    return nn.Parameter(
        torch.FloatTensor(embeddings),
        requires_grad=False
    )


@dataclass
class ModelArgs:
    LOG_DIR: str  # Directory containing model checkpoints and embeddings
    vocab_size: int = -1 # later loaded from tokenizer
    n_embd: int = 576
    n_embd_factor: int = None
    hidden_dim: int = 768
    n_layers: int = 6
    n_ctx: int = 2048
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    drop_out: float = 0.1
    hidden_act: str = "silu"
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 0
    eos_token_id: int = 0
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    use_hierarchy_embd: bool = False
    hierarchy_dim: Optional[int] = None
    use_semantic_embd: bool = False
    semantic_dim: Optional[int] = None
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})
    
    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def precompute_rope_frequencies(n_embd: int, n_ctx: int, theta: float = 10000.0):
    position = torch.arange(n_ctx).unsqueeze(1)  # [seq_len, 1]
    div_term = theta ** (torch.arange(0, n_embd, 2).float() / n_embd)   # [n_embd/2]
    freqs = position / div_term  # [seq_len, n_embd/2]
    return freqs

def apply_rotary_embeddings(x: torch.Tensor, freqs: torch.Tensor):
    # x shape: [batch, seq_len, heads, head_dim]
    # freqs shape: [seq_len, head_dim/2]
    x_rot = x.float()
    
    # Reshape freqs to match x's dimensions
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, n_embd/2]
    
    # Split channels for rotation
    x1, x2 = x_rot[..., :x_rot.shape[-1]//2], x_rot[..., x_rot.shape[-1]//2:]
    
    # Apply rotary embeddings
    cos = torch.cos(freqs).to(x.device)
    sin = torch.sin(freqs).to(x.device)
    
    # Ensure broadcasting dimensions match
    cos = cos.expand_as(x1)
    sin = sin.expand_as(x1)
    
    # Rotate x1 and x2
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin
    
    # Concatenate back
    return torch.cat([x1_rot, x2_rot], dim=-1).to(x.dtype)

def apply_rope_with_pos_ids(x: torch.Tensor, freqs: torch.Tensor, position_ids: torch.Tensor):
    """
    x: [B, T, H, Dh]  (queries or keys)
    freqs: [max_seq_len, Dh/2]  (precomputed table)
    position_ids: [B, T] absolute positions for these tokens
    """
    B, T, H, Dh = x.shape
    x = x.float()

    # gather the cos/sin rows for each position in the batch
    cos = torch.cos(freqs[position_ids])  # [B, T, Dh/2]
    sin = torch.sin(freqs[position_ids])  # [B, T, Dh/2]

    # expand to heads
    cos = cos.unsqueeze(2).expand(B, T, H, Dh // 2)  # [B, T, H, Dh/2]
    sin = sin.unsqueeze(2).expand(B, T, H, Dh // 2)

    x1, x2 = x[..., :Dh//2], x[..., Dh//2:]
    x_rot1 = x1 * cos - x2 * sin
    x_rot2 = x2 * cos + x1 * sin
    out = torch.cat([x_rot1, x_rot2], dim=-1)
    return out.to(dtype=x.dtype)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_embd = args.n_embd
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.n_embd // args.num_attention_heads
        
        # Adjust projections to match head dimensions
        self.q_proj = nn.Linear(args.n_embd, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.n_embd, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.n_embd, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, args.n_embd, bias=False)
        
        # Initialize rotary embeddings
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                self.head_dim,  # Use full head_dim for frequencies
                args.n_ctx,
                args.rope_theta
            ),
            persistent=False
        )
        self.attn_drop = nn.Dropout(args.drop_out)
        self.residual_drop = nn.Dropout(args.drop_out)

    def forward(self, hidden_states, attention_mask=None, past_key_value: Optional[tuple] = None, use_cache: bool = False, position_offset: Optional[int] = None):
        """
        hidden_states: [B, Tq, D]
        past_key_value: Optional[(past_k, past_v)] each [B, H_kv, Tp, Dh] already RoPE-rotated
        use_cache: if True, return current (k,v) concatenated with past for caching upstream
        position_offset: if provided, absolute position of the first token in `hidden_states`
                        (i.e., past length). If None, defaults to 0.
        """
        B, Tq, _ = hidden_states.size()
        Hq = self.num_heads
        Hkv = self.num_kv_heads
        Dh = self.head_dim
        
        # Projections
        q = self.q_proj(hidden_states).view(B, Tq, Hq, Dh)
        k_new = self.k_proj(hidden_states).view(B, Tq, Hkv, Dh)
        v_new = self.v_proj(hidden_states).view(B, Tq, Hkv, Dh)
        
        # Absolute positions for the NEW tokens
        past_len = 0 if past_key_value is None else past_key_value[0].size(2)
        if position_offset is not None:
            position_offset = past_len
        pos_ids_new = (torch.arange(Tq, device=hidden_states.device) + position_offset).view(1, Tq).expand(B, Tq)
        
        # Apply RoPE to new q and k
        q = apply_rope_with_pos_ids(q, self.rope_freqs, pos_ids_new)
        k_new = apply_rope_with_pos_ids(k_new, self.rope_freqs, pos_ids_new)
        
        # Prepare full K/V in KV-head space, concatenate with past if any
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_cat = torch.cat([past_k, k_new.transpose(1, 2)], dim=2)
            v_cat = torch.cat([past_v, v_new.transpose(1, 2)], dim=2)
        else:
            k_cat = k_new.transpose(1, 2)
            v_cat = v_new.transpose(1, 2)
        Tk = k_cat.size(2)
        
        # Expand KV to query-heads if using GQA
        if Hkv < Hq:
            repeat = Hq // Hkv
            k_full = k_cat.repeat_interleave(repeat, dim=1)
            v_full = v_cat.repeat_interleave(repeat, dim=1)
        else:
            k_full = k_cat
            v_full = v_cat
        
        # Scaled dot-product attention
        q = q.transpose(1, 2)  # (B, Hq, Tq, Dh)
        
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(Dh)
        
        i_abs = (position_offset + torch.arange(Tq, device=hidden_states.device).unsqueeze(1))
        j_abs = torch.arange(Tk, device=hidden_states.device).unsqueeze(0)
        causal = (j_abs <= i_abs).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~causal, float('-inf'))
        
         # Optional extra mask (e.g., padding mask shaped/broadcastable to [B,1,Tq,Tk])
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
                
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        context = torch.matmul(attn_probs, v_full)

        context = context.transpose(1, 2).contiguous().view(B, Tq, Hq*Dh)
        out = self.o_proj(context)
        out = self.residual_drop(out)
        if use_cache:
            return out, (k_cat, v_cat)
        else:
            return out, None

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.n_embd, args.hidden_dim, bias=False)
        self.up_proj = nn.Linear(args.n_embd, args.hidden_dim, bias=False)
        self.down_proj = nn.Linear(args.hidden_dim, args.n_embd, bias=False)
        self.act_fn = nn.SiLU()
        self.drop_out = nn.Dropout(args.drop_out)
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.drop_out(out)

class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = SelfAttention(args)
        self.ffn = FeedForward(args)
        self.input_layernorm = RMSNorm(args.n_embd, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.n_embd, args.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_value: Optional[tuple] = None, use_cache: bool = False, position_offset: Optional[int] = None):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(x, attention_mask, past_key_value, use_cache, position_offset)
        hidden_states = residual + attn_out
        
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        ffn_out = self.ffn(x)
        hidden_states = residual + ffn_out
        
        return hidden_states, present_kv

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        if args.n_embd_factor is not None:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.n_embd_factor)
            self.emb_proj = nn.Linear(args.n_embd_factor, args.n_embd, bias=False)
        else:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.n_embd)
        self.emb_drop = nn.Dropout(args.drop_out)
        # Optional auxiliary embeddings
        if args.use_hierarchy_embd:
            hierarchy_embd_path = os.path.join(args.LOG_DIR, "knowledge_embd", "hierarchy_embd.npz")
            self.hierarchy_embeddings = load_auxiliary_embeddings(
                hierarchy_embd_path,
                args.vocab_size,
                args.hierarchy_dim,
                "hierarchy"
            )
            self.hierarchy_proj = nn.Linear(args.hierarchy_dim, args.n_embd, bias=False)  # project to n_embd
            self.alpha_h = nn.Parameter(torch.tensor(1.0))
            self.h_rms = RMSNorm(args.n_embd, args.rms_norm_eps)
        if args.use_semantic_embd:
            semantic_embd_path = os.path.join(args.LOG_DIR, "knowledge_embd", "semantic_embd.npz")
            self.semantic_embeddings = load_auxiliary_embeddings(
                semantic_embd_path,
                args.vocab_size,
                args.semantic_dim,
                "semantic"
            )
            self.semantic_proj = nn.Linear(args.semantic_dim, args.n_embd, bias=False)  # project to n_embd
            self.alpha_s = nn.Parameter(torch.tensor(1.0))
            self.s_rms = RMSNorm(args.n_embd, args.rms_norm_eps)
            
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args))
        self.final_norm = RMSNorm(args.n_embd, args.rms_norm_eps)
        # Add output before weight tying
        self.output = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        # Initialize weights
        self.apply(self._init_weights)
        if args.n_embd_factor is not None:
            self.output_proj = nn.Linear(args.n_embd, args.n_embd_factor, bias=False)
            self.output = nn.Linear(args.n_embd_factor, args.vocab_size, bias=False)
            if args.tie_word_embeddings:
                self.output.weight = self.tok_embeddings.weight
        else:
            self.output = nn.Linear(args.n_embd, args.vocab_size, bias=False)
            if args.tie_word_embeddings:
                self.output.weight = self.tok_embeddings.weight
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.args.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.args.initializer_range)

    def print_model_params(self, print_details: bool = True):
        """Print a detailed breakdown of model parameters."""
        total_params = 0
        if print_details:
            print("\nModel Parameter Details:")
            print("-" * 100)
            print(f"{'Layer':<40} {'Shape':>20} {'Parameters':>15} {'Status':>15}")
            print("-" * 100)
            
            for name, param in self.named_parameters():
                param_count = param.numel()
                total_params += param_count
                status = "Trainable" if param.requires_grad else "Frozen"
                print(f"{name:<40} {str(list(param.shape)):>20} {param_count:>15,} {status:>15}")
            
        print("-" * 80)
        print(f"{'Total Parameters':<40} {' ':>20} {total_params:>15,}")
        print("\nParameter count by component:")
        
        # Count parameters by major components
        def count_params(pattern):
            all_params = sum(p.numel() for name, p in self.named_parameters() if pattern in name)
            trainable = sum(p.numel() for name, p in self.named_parameters() if pattern in name and p.requires_grad)
            frozen = sum(p.numel() for name, p in self.named_parameters() if pattern in name and not p.requires_grad)
            return all_params, trainable, frozen

        components = {
            'Token Embeddings': 'tok_embeddings',
            'Hierarchy Embeddings': 'hierarchy_embeddings',
            'Hierarchy Projection': 'hierarchy_proj',
            'Semantic Embeddings': 'semantic_embeddings',
            'Semantic Projection': 'semantic_proj',
            'DecoderBlock': 'layers',
            'Final Norm': 'final_norm',
            'Output Layer': 'output'
        }
        
        print(f"{'Component':<20} {'Total':>15} {'Trainable':>15} {'Frozen':>15}")
        print("-" * 70)
        
        for component, pattern in components.items():
            total, trainable, frozen = count_params(pattern)
            print(f"{component:<20} {total:>15,} {trainable:>15,} {frozen:>15,}")
        
        # Count trainable vs non-trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"\nTrainable parameters:   {trainable_params:>15,}")
        print(f"Frozen parameters:      {frozen_params:>15,}")
        print(f"Total parameters:      {trainable_params + frozen_params:>15,}")
    
    def forward(self, input_ids, attention_mask=None, past_key_values: Optional[list] = None, use_cache: bool = False):
        B, Tq = input_ids.shape
        device = input_ids.device
        if self.args.n_embd_factor is not None:
            hidden_states = self.emb_proj(self.tok_embeddings(input_ids))  # [B, Tq, D]
        else:
            hidden_states = self.tok_embeddings(input_ids)  # [B, Tq, D]
        
        # Optional auxiliary embeddings (unchanged)
        if self.args.use_hierarchy_embd:
            h = self.hierarchy_proj(self.hierarchy_embeddings[input_ids])
            hidden_states = hidden_states + self.alpha_h * self.h_rms(h)
        if self.args.use_semantic_embd:
            s = self.semantic_proj(self.semantic_embeddings[input_ids])
            hidden_states = hidden_states + self.alpha_s * self.s_rms(s)
        hidden_states = self.emb_drop(hidden_states)
        # We build causal mask per layer inside attention using absolute positions. If need padding masks, use the attention_mask argument.
        
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Absolute starting position for FIRST query token in this call
        past_len = 0 if past_key_values[0] is None else past_key_values[0][0].size(2)
        
        presents = [] if use_cache else None
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(hidden_states, attention_mask, past_kv, use_cache, position_offset=past_len)
            if use_cache:
                presents.append(present_kv)
        hidden_states = self.final_norm(hidden_states)
        if self.args.n_embd_factor is not None:
            hidden_states = self.output_proj(hidden_states)
        logits = self.output(hidden_states)
        if use_cache:
            return logits, presents
        else:
            return logits
    
    @staticmethod
    def _sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """
        logits: [B, V]
        returns: next_token ids [B]
        """
        if temperature <= 0:
            # greedy fallback
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)                      # [B, V]

        # sort by prob desc
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # [B, V], [B, V]
        cumsum = torch.cumsum(sorted_probs, dim=-1)                               # [B, V]

        # mask everything past the nucleus (keep the first token that crosses top_p)
        cutoff = cumsum > top_p                                                   # [B, V] boolean
        # shift mask right so we keep at least one token per row
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        # re-normalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # sample in the sorted space then map back to original vocab ids
        next_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)          # [B, 1]
        next_token = torch.gather(sorted_idx, -1, next_sorted_idx).squeeze(-1)    # [B]
        return next_token

    def print_alpha_values(self):
        alpha_h, alpha_s = None, None
        if hasattr(self, 'alpha_h'):
            print(f"Hierarchy (alpha_h): {self.alpha_h.item():.4f}")
            alpha_h = self.alpha_h.item()
        if hasattr(self, 'alpha_s'):
            print(f"Semantic (alpha_s): {self.alpha_s.item():.4f}")
            alpha_s = self.alpha_s.item()
        return alpha_h, alpha_s

    def generate(self, input_ids, max_length=None, temperature=1.0, top_p=0.9, end_token_id=None, tokenizer=None):
        """
        input_ids: [B, T] (B can be 1)
        tokenizer: LocalTokenizer instance for constrained decoding
        returns:   [B, T_out]
        """
        self.eval()
        max_length = self.args.n_ctx if max_length is None else min(max_length, self.args.n_ctx)
        end_token_id = self.args.eos_token_id if end_token_id is None else end_token_id # default is eos_token_id, might be different for different tasks
        
        device = input_ids.device
        cur = input_ids
        B = cur.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Prepare Q token IDs for masking if constrained decoding is enabled
        q_token_ids = None
        if tokenizer is not None:
            q_token_ids = torch.tensor(tokenizer.get_q_token_ids(), device=device)
        
        with torch.no_grad():
            logits, past = self(cur, use_cache=True)
            
            while cur.size(1) < max_length:
                next_logits = logits[:, -1, :].clone() # [B, V]
                
                # Apply constrained decoding if enabled
                if tokenizer is not None:
                    # Get the last generated token for each batch
                    last_token = cur[:, -1]  # [B]
                    
                    # For each sample in batch, check if last token is LAB token
                    for b in range(B):
                        if finished[b]:
                            continue
                            
                        is_last_lab = tokenizer.is_lab_token(last_token[b])
                        
                        if is_last_lab:
                            # Last token is LAB_xxx: encourage Q tokens
                            mask = torch.ones_like(next_logits[b], dtype=torch.bool)
                            mask[q_token_ids] = False
                            next_logits[b, mask] = float('-inf')
                        else:
                            # Last token is NOT LAB_xxx: mask out all Q tokens
                            next_logits[b, q_token_ids] = float('-inf')
                else:
                    raise ValueError("No tokenizer provided for constrained decoding")
                next_token = Transformer._sample_top_p(next_logits, top_p, temperature) # [B]
                next_token = torch.where(finished, torch.full_like(next_token, end_token_id), next_token)
                
                cur = torch.cat([cur, next_token.unsqueeze(1)], dim=1) # [B, T+1] 
                finished = finished | (next_token == end_token_id)
                if torch.all(finished):
                    break
                
                last = next_token.view(B, 1)
                logits, past = self(last, use_cache=True, past_key_values=past)
                    
        return cur 