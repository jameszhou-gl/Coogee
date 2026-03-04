import torch

class LocalTokenizer:
    """Simple tokenizer that loads from local vocab.csv"""
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.id_to_token = {}
        self.token_to_id = {}
        
        # Load vocab
        with open(vocab_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                if line.strip():
                    token_id, concept_code = line.strip().split(',', 1)
                    token_id = int(token_id)
                    self.id_to_token[token_id] = concept_code
                    self.token_to_id[concept_code] = token_id
        
        self.vocab_size = len(self.id_to_token)
        
        # Set special tokens based on actual vocab
        # START_RECORD: 35139, END_RECORD: 1185, PADDING: 35130
        self.bos_token_id = self.token_to_id.get("START_RECORD", 35139)
        self.eos_token_id = self.token_to_id.get("END_RECORD", 1185)
        self.pad_token_id = self.token_to_id.get("PADDING", 35130)
        self.unk_token_id = None  # No explicit UNK token in vocab
        
        print(f"[LocalTokenizer] Loaded {self.vocab_size} tokens from {vocab_path}")
        print(f"[LocalTokenizer] Special tokens: BOS={self.bos_token_id}, EOS={self.eos_token_id}, PAD={self.pad_token_id}")
    
    def decode(self, token_ids):
        """Decode a list of token IDs to a string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = [self.id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]
        return " | ".join(tokens)
    
    def decode_token_id(self, token_id):
        """Decode a single token ID to a string."""
        return self.id_to_token.get(token_id, f"<UNK_{token_id}>")
    
    def convert_tokens_to_ids(self, token):
        """Convert a token string to its ID. Returns None if not found."""
        return self.token_to_id.get(token, None)
    
    def is_lab_token(self, token_id):
        """Check if a token ID corresponds to a LAB_xxx token."""
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        token_str = self.id_to_token.get(token_id, "")
        return token_str.startswith("LAB_")
    
    def is_q_token(self, token_id):
        """Check if a token ID corresponds to a _Q quantile token."""
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        token_str = self.id_to_token.get(token_id, "")
        return token_str.startswith("_Q") and len(token_str) <= 4  # _Q1 to _Q10
    
    def get_q_token_ids(self):
        """Get all quantile token IDs (_Q1 through _Q10)."""
        q_ids = []
        for token_id, token_str in self.id_to_token.items():
            if self.is_q_token(token_id):
                q_ids.append(token_id)
        return sorted(q_ids)