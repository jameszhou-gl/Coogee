import argparse
import csv
import os

from typing import Dict, List
from huggingface_hub import HfApi

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# todo: push tokenizer and model to hf
def read_vocab_csv(vocab_csv: str) -> Dict[str, int]:
    """Read vocab.csv -> {token_string: token_id}."""
    vocab: Dict[str, int] = {}
    with open(vocab_csv, newline="") as f:
        reader = csv.DictReader(f)
        assert {"token_id", "concept_code"} <= set(reader.fieldnames or []), \
            "vocab.csv must have columns: token_id, concept_code"
        for row in reader:
            tok_id = int(row["token_id"])
            tok_str = str(row["concept_code"])
            if tok_str in vocab:
                raise ValueError(f"Duplicate token string in vocab: {tok_str}")
            vocab[tok_str] = tok_id
    # Sanity: IDs are 0..N-1 with no gaps (HF prefers contiguous ids)
    ids = sorted(vocab.values())
    if ids != list(range(len(ids))):
        raise ValueError("token_id must be contiguous from 0..N-1")
    return vocab


def infer_additional_special_tokens(vocab: Dict[str, int],
                                    bos: str, eos: str, pad: str | None) -> List[str]:
    """Pick domain special tokens (e.g., visit markers) to register as 'additional_special_tokens'."""
    candidates = []
    for t in vocab.keys():
        if t in {bos, eos}:
            continue
        if pad and t == pad:
            continue
        # Heuristics: register common structural markers as special
        if any(tag in t for tag in ["START_VISIT", "END_VISIT"]):
            candidates.append(t)
    return sorted(set(candidates))


def build_tokenizer(vocab: Dict[str, int],
                    bos: str, eos: str, pad: str | None,
                    add_specials: List[str]) -> PreTrainedTokenizerFast:
    """
    Create a WordLevel tokenizer:
    - No subword split; each string maps 1:1 to an id from vocab.csv.
    - Adds a reserve [UNK] if not present (never used by your pipeline but required by WordLevel).
    """
    # WordLevel requires an unk token; add a reserved one if missing.
    unk_token = "[UNK]"
    if unk_token not in vocab:
        vocab = {**vocab, unk_token: len(vocab)}

    # Build low-level tokenizer
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))
    tok.pre_tokenizer = Whitespace()  # (kept for completeness; we use is_split_into_words=True)

    # Wrap as HF fast tokenizer
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=bos,
        eos_token=eos,
        unk_token=unk_token,
        pad_token=(pad if pad else None),
        additional_special_tokens=add_specials,
    )

    # Ensure BOS/EOS are actually added on encode if desired (HF respects these flags)
    hf_tok.add_bos_token = True
    hf_tok.add_eos_token = True

    return hf_tok

def push_to_hf(out_dir: str, repo: str):
    """
    Push the tokenizer to Hugging Face.
    """
    open(os.path.join(out_dir, "README.md"), "w").write(
        "# EHR Tokenizer\n\nWord-level tokenizer for EHR tokens (concept codes & structure markers)."
    )

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo,
        repo_type="model",
    )
    print(f"Uploaded to https://huggingface.co/{repo}")
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--LOG_DIR", default="artifacts/mimiciv_2.2_meds_full")
    ap.add_argument("--bos", default="START_RECORD")
    ap.add_argument("--eos", default="END_RECORD")
    ap.add_argument("--pad", default="PADDING",
                    help="Set if you use padding mode (e.g., PADDING); omit for packed mode.")
    args = ap.parse_args()
    tokenizer_dir = os.path.join(args.LOG_DIR, "tokenizer")
        
    vocab = read_vocab_csv(os.path.join(tokenizer_dir, "vocab.csv"))

    # Validate special tokens exist (except pad if None)
    for name, tok in [("BOS", args.bos), ("EOS", args.eos)]:
        if tok not in vocab:
            raise ValueError(f"{name} token '{tok}' not found in vocab.csv")
    if args.pad and args.pad not in vocab:
        raise ValueError(f"PAD token '{args.pad}' not found in vocab.csv")

    additional = infer_additional_special_tokens(vocab, args.bos, args.eos, args.pad)
    tokenizer = build_tokenizer(vocab, args.bos, args.eos, args.pad, additional)

    # Save in HF format (creates tokenizer.json + tokenizer_config.json + special_tokens_map.json)
    tokenizer.save_pretrained(tokenizer_dir)

    # Quick self-test & example
    example = ["START_RECORD", "START_VISIT", "END_VISIT", "END_RECORD"]
    enc = tokenizer(
        example,
        is_split_into_words=True,
        add_special_tokens=True,   # adds BOS/EOS if configured above
        return_tensors=None,
    )
    print(f"[OK] Saved tokenizer to: {tokenizer_dir}")
    print("Example encode (ids):", enc["input_ids"])
    print("Specials:",
          {"bos": tokenizer.bos_token, "eos": tokenizer.eos_token,
           "pad": tokenizer.pad_token, "additional": tokenizer.additional_special_tokens})

    push_to_hf(tokenizer_dir, repo="jameszhou-gl/ehr-gpt")
if __name__ == "__main__":
    main()
    
# python scripts/push_to_hf.py 