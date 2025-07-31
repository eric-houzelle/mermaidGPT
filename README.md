---
language: en
license: mit
tags:
  - mermaid
  - diagram-generation
  - gpt
  - natural-language-processing
  - english
  - text-to-diagram
model-index:
  - name: mermaidGPT
    results: []
---

# 🧠 mermaidGPT — Générateur de graphes Mermaid à partir de texte

**mermaidGPT** est un petit modèle en français capable de générer des **diagrammes Mermaid.js** à partir d'instructions en langage naturel. Il permet de transformer des phrases simples comme "Créer un diagramme de flux montrant un utilisateur qui se connecte à un site" en code Mermaid valide.

---

## 🎯 Objectif

Ce projet vise à faciliter la création de diagrammes techniques à partir de phrases naturelles, pour des usages pédagogiques, documentaires ou de prototypage rapide.

---

## 🧰 Fonctionnalités

- Prise en charge du **français**
- Support des **diagrammes de flux**, **organigrammes**, **diagrammes de séquence**, etc.
- Modèle léger et rapide à exécuter
- Export en **.mmd** (fichier texte Mermaid) ou en **image** via rendu
- Interface CLI ou API (selon l’implémentation)

---

## 🗃️ Structure du projet

\`\`\`
/
├── generate.py         # Génération de diagrammes à partir d'une phrase
├── model.py            # Architecture du modèle
├── tokenizer.py        # Tokenizer et prétraitement du texte
├── data/               # Données d'entraînement (optionnelles ou mock)
├── examples/           # Exemples de phrases et de sorties Mermaid
├── checkpoints/        # Modèles sauvegardés
└── README.md
\`\`\`

---

## 🚀 Exemple d'utilisation

### Entrée :
\`\`\`
Créer un diagramme de flux montrant un utilisateur qui envoie une requête à un serveur, et reçoit une réponse.
\`\`\`

### Sortie (code Mermaid) :
\`\`\`mermaid
graph TD
    Utilisateur -->|Requête| Serveur
    Serveur -->|Réponse| Utilisateur
\`\`\`

---

## 📦 Installation

\`\`\`bash
git clone https://github.com/eric-houzelle/mermaidGPT.git
cd mermaidGPT
pip install -r requirements.txt
\`\`\`

---

## 🧪 Entraînement

\`\`\`bash
python train.py
\`\`\`

---

## ✍️ Génération

\`\`\`bash
python generate.py --prompt "Créer un diagramme montrant A qui appelle B, puis B répond à A"
\`\`\`

---

## 🧠 Modèle utilisé

Le modèle repose sur une architecture GPT-like entraînée à générer du code Mermaid à partir d'un prompt en langage naturel.

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier \`LICENSE\`.

---

## 🤗 Hébergement Hugging Face

Le modèle peut être testé en ligne sur [Hugging Face Spaces](https://huggingface.co/spaces/) ou téléchargé depuis :  
📦 https://huggingface.co/eric-houzelle/mermaidGPT

---

## ✉️ Co
