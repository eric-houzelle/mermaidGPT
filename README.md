---
language: en
license: mit
tags:
  - gpt
  - diagram-generation
  - mermaid
  - text-to-diagram
  - natural-language-processing
  - toy-model
datasets:
  - custom
model-index:
  - name: mermaidGPT
    results: []
---

# 🧠 mermaidGPT — Mermaid Diagram Generator from Natural Language

**mermaidGPT** is a small English-language model trained to generate valid [Mermaid.js](https://mermaid-js.github.io/) diagrams from natural language prompts. It allows users to create flowcharts, sequence diagrams, and more using simple instructions like “Show a user sending a request to a server.”
Build with the dataset : [Celiadraw/text-to-mermaid](https://huggingface.co/datasets/Celiadraw/text-to-mermaid)

---

## 🧰 Features

- English-only natural language input
- Generates Mermaid.js graph code (flowcharts, sequence diagrams, etc.)
- GPT-like architecture using PyTorch
- Lightweight and fast (trainable on a single GPU)
- CLI-based usage for generation
- Can be extended with API or frontend

---

## 📦 Project Structure

\`\`\`
/
├── train.py              # Training script
├── generate.py           # Generate Mermaid code from prompt
├── tokenizer.py          # Tokenization utilities
├── model.py              # GPT-style model architecture
├── data/                 # Training data (if any)
├── examples/             # Prompt examples and output
├── checkpoints/          # Saved model checkpoints
└── README.md
\`\`\`

---

## 🚀 Training

### 1. Build the tokenizer

\`\`\`bash
python tokenizer.py
\`\`\`

---

### 2. Train the model

\`\`\`bash
python train.py
\`\`\`

---

### 3. Hyperparameters

\`\`\`
block_size = 128
batch_size = 32
learning_rate = 1e-3
embed_dim = 128
n_heads = 8
n_layers = 8
\`\`\`

---

## ✍️ Generation Example

### Prompt:
\`\`\`
Create a flowchart showing a user sending a request to a server and receiving a response.
\`\`\`

### Output:
\`\`\`mermaid
graph TD
    User -->|Request| Server
    Server -->|Response| User
\`\`\`

---

## 📄 License

This project is released under the MIT License. See the \`LICENSE\` file for details.

---

## 🤗 Model on Hugging Face

Test or download the model on Hugging Face:  
📦 https://huggingface.co/eric-houzelle/mermaidGPT

---

## 📬 Contact

For any suggestions or questions: \`eric.houzelle@email.com\`
