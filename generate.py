import torch
from model import MiniGPT
from tokenizer import load_tokenizer
import argparse

def render_mermaid_to_html(mermaid_code, output_path="diagram.html"):
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true }});
  </script>
</head>
<body>
  <pre class="mermaid">
{mermaid_code}
  </pre>
</body>
</html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Diagramme Mermaid sauvegardé dans : {output_path}")

# ------- Paramètres -------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "checkpoints/model_step_best.pt"  # ← remplace par ton fichier
tokenizer_path = "tokenizer_mermaid.json"
block_size = 256       # taille du contexte, voir plus loin dans la phrase
batch_size = 16       # nombre de séquences par batch
embed_dim = 256
n_heads = 16 
n_layers = 16
max_new_tokens = 150

# ------- Load tokenizer -------
stoi, itos, encode, decode, pad_token_id, eos_token_id = load_tokenizer(tokenizer_path)
vocab_size = len(stoi)

# ------- Load model -------
model = MiniGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=embed_dim,
    depth=n_layers,
    heads=n_heads
).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------- Contexte initial -------
prompt = "Design a sequence diagram for an online shopping checkout process. Include actors such as User, Shopping Cart, Payment Gateway, and illustrate interactions like Add to Cart, Proceed to Checkout, and Complete Payment.<|sep|>"
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Texte à transformer en diagramme Mermaid")
args = parser.parse_args()

prompt = args.prompt.strip()
prompt = prompt + "<|sep|>"
#prompt = "Create a flowchart for a hiring process in a corporate environment. Include steps like 'Job Posting', 'Resume Screening', 'Interview Scheduling', 'Interviews', 'Reference Checks', 'Offer Preparation', and 'Hiring Decision'.<|sep|>"
context_ids = encode(prompt)
if isinstance(context_ids, torch.Tensor):
    context = context_ids.unsqueeze(0).to(device) 
else:
    context = torch.tensor([context_ids], dtype=torch.long, device=device)


# ------- Génération -------
with torch.no_grad():
    output_ids = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=0.7,      
        top_k=50               
    )[0].tolist()

# ------- Décodage -------
generated_text = decode(output_ids[len(context_ids):])
print("\n--- Generated diagram ---\n")
print(generated_text)
print("\n------------------------\n")
render_mermaid_to_html(generated_text)