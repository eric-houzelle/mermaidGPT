# mini_gpt_transformer/train.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import MiniGPT
from datasets import load_dataset
from dataloader import MermaidDataset
from torch.utils.data import DataLoader
import os
from torch.nn.utils.rnn import pad_sequence
from tokenizer import load_tokenizer
from utils import print_gpu_memory
import time
from torch.optim.lr_scheduler import OneCycleLR
import trackio
import random
import time




# ----------- Hyperparamètres -----------
block_size = 256       # taille du contexte, voir plus loin dans la phrase
batch_size = 16       # nombre de séquences par batch
max_iters = 1000000       # nombre d'itérations d'entraînement
eval_interval = 100   # fréquence d'évaluation
learning_rate = 1e-4 # 5e-5
embed_dim = 256
n_heads = 16 
n_layers = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dt = load_dataset("Celiadraw/text-to-mermaid")
texts = [f"{item['prompt']}<|sep|>{item['output']} <eos>" for item in dt["train"]]

stoi, itos, encode, decode, pad_token_id, eos_token_id = load_tokenizer("tokenizer_mermaid.json")
vocab_size = len(stoi) 


resume_path = "checkpoints/model_step_best.pt" 
if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    start_iter = checkpoint["step"] + 1
    print(f"Reprise à l'étape {start_iter}")
else:
    start_iter = 0
# ---------- Création du modèle une fois vocab prêt ----------
model = MiniGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=embed_dim,
    depth=n_layers,
    heads=n_heads
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------- Puis chargement des poids si reprise ----------
if os.path.exists(resume_path):
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



def collate_fn(batch):
    xs, ys = zip(*batch) 
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=pad_token_id)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=pad_token_id)
    return xs_padded, ys_padded



split_idx = int(0.9 * len(texts))
train_texts = texts[:split_idx]
val_texts = texts[split_idx:]

train_dataset = MermaidDataset(train_texts, block_size, encode)
val_dataset = MermaidDataset(val_texts, block_size, encode)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1e9:
        return f"{total/1e9:.2f}B"
    elif total >= 1e6:
        return f"{total/1e6:.2f}M"
    elif total >= 1e3:
        return f"{total/1e3:.2f}K"
    return str(total)

print("Nombre de paramètres du modèle :", count_parameters(model))


# ----------- Learning rate scheduler -----------
scheduler = OneCycleLR(
    optimizer,
    max_lr=learning_rate,      
    total_steps=max_iters,     
)


# ----------- Boucle d'entraînement -----------
num_epochs = 100  
global_step = start_iter 
best_loss = 10000
trackio.init(project="mermaidGPT", config={
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "num_heads": n_heads,
            "num_layers": n_layers
        })
for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
    val_loss_total = 0
    for xb, yb in train_loader:
        start_time_total = time.time()
        xb = xb.to(device)
        yb = yb.to(device)
        model.train()
        #print_gpu_memory("Train ")
        
        start_time = time.time()
        logits = model(xb)
        forward_time = time.time() - start_time
        
        #print_gpu_memory("Logits")
        
        start_time = time.time()
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T), ignore_index=pad_token_id)
        val_loss_total += loss.item()
        loss_time = time.time() - start_time
        preds = logits.argmax(dim=-1)
        correct = (preds == yb) & (yb != pad_token_id)
        accuracy = correct.sum().item() / (yb != pad_token_id).sum().item()
        

        trackio.log({
            "loss": loss.item(),
            "accuracy": accuracy
        })
        
        #print_gpu_memory("Loss  ")
        
        start_time = time.time()
        optimizer.zero_grad()
        #print_gpu_memory("Zero G")
        loss.backward()
        backward_time = time.time() - start_time
        
        #print_gpu_memory("Back w")
        
        start_time = time.time()
        optimizer.step()
        scheduler.step()
        step_time = time.time() - start_time
        
        #print_gpu_memory("Opt st")
        end_time_total = time.time()
        
        total_time = time.time() - start_time_total
        print(f"[Step {global_step}] Perte = {loss.item():.4f} | total: {total_time:.3f}s | forward: {forward_time:.3f}s | loss: {loss_time:.3f}s | backward: {backward_time:.3f}s | step: {step_time:.3f}s")


        
        if global_step % eval_interval == 0:
            print(f"[Epoch {epoch+1} | Step {global_step}] Perte = {loss.item():.4f}")
            model.eval()
            prompt = "Design a sequence diagram for an online shopping checkout process. Include actors such as User, Shopping Cart, Payment Gateway, and illustrate interactions like Add to Cart, Proceed to Checkout, and Complete Payment.<|sep|>"
            ids = encode(prompt)
            if isinstance(ids, torch.Tensor):
                context = ids.unsqueeze(0).to(device) 
            else:
                context = torch.tensor([ids], dtype=torch.long, device=device)

            generated = model.generate(context, max_new_tokens=150, eos_token_id=eos_token_id)[0]
            print("\n--- Généré ---")
            print(decode(generated.tolist()))
            print("--------------\n")
        else:
            print(f"[Epoch {epoch+1} | Step {global_step}] Perte = {loss.item():.4f}")


        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'vocab': {'stoi': stoi, 'itos': itos}
            }, f"checkpoints/model_step_best.pt")

        global_step += 1
    val_loss_avg = val_loss_total / len(val_loader)

    trackio.log({
        "val_loss": val_loss_avg,
    })
    
    

trackio.finish()