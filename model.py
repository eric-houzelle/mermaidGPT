# mini_gpt_transformer/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloc Self-Attention : calcule l'attention entre les tokens de la séquence
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        # Projette l'entrée en 3 vecteurs : requête, clé, valeur (Q, K, V)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        # Sépare Q, K, V et prépare pour multi-têtes : (3, B, heads, T, head_dim)
        qkv = qkv.reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Produit scalaire QK^T, puis normalisation
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # masque causal
        weights = F.softmax(scores, dim=-1)  # pondération
        attn = weights @ v  # combinaison pondérée des valeurs

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)  # fusion têtes
        return self.out(attn)  # projection de sortie

# Bloc Transformer : attention + normalisation + MLP
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # couche cachée élargie
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),  # retour à embed_dim
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))  # résiduel + attention
        x = x + self.dropout(self.ff(self.ln2(x)))          # résiduel + MLP
        return x

# Modèle MiniGPT complet
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim=128, depth=4, heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)  # embeddings des tokens, chaque entrée du tokenizer a un vecteur associé qui represente ses caracteritiques
        self.pos_emb = nn.Embedding(block_size, embed_dim)    # embeddings positionnels pour savoir ou se trouve chaque token dans le block
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)]) # on empile les couches neuronales
        self.ln_f = nn.LayerNorm(embed_dim)                   # normalisation finale pour faciliter la convergence et les valeurs trop grandes 
        self.head = nn.Linear(embed_dim, vocab_size)          # prédiction des logits vocab
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape # idx c'est une matrice de vecteurs de taille du batch et de block_size, donc B correspond  à la taille du batch et T la taille du block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # créé le vecteurs de position, de 0 a T [0,1,2,3,4,5,6,...] et unqsueeze : [[0,1,2,3,4,5,6,...]]
        x = self.token_emb(idx) + self.pos_emb(pos)  # addition token + position

        # masque causal : triangle inférieur T x T
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0) # Produit un masque causal (une matrice avec des 1 sous la diagonale) de shape (1, 1, T, T), utilisé pour s'assurer que le modèle ne regarde que le passé et le présent, jamais le futur.
        for block in self.blocks:
            x = block(x, mask)  # passe à travers les blocs Transformer
        x = self.ln_f(x)          # normalisation finale
        logits = self.head(x)     # projection vers le vocabulaire
        return logits

    # Génération autoregressive : ajoute des tokens un par un
    def generate(
        self, idx, max_new_tokens, eos_token_id=None,
        temperature=1.0, top_k=None
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # ← temp. control

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -float("Inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        return idx