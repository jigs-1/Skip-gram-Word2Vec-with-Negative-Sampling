# Skip-Gram Word2Vec with Negative Sampling

This repository contains an implementation of the Skip-Gram Word2Vec model with Negative Sampling, developed as part of an academic assignment for Natural Language Processing.

The model is trained on a cleaned Wikipedia corpus (enwik8) and learns dense word embeddings that capture semantic relationships between words. The implementation follows the core ideas proposed by Mikolov et al. (2013) and focuses on clarity and learning rather than large-scale optimization.

---

## Project Overview

The objective of this project is to:
- Implement the Skip-Gram architecture with Negative Sampling from scratch
- Train word embeddings on a Wikipedia-based dataset
- Evaluate the learned embeddings using similarity, analogy, and bias detection tasks

This project is intended for educational and academic purposes.

---

## Key Features

- Skip-Gram with Negative Sampling
- Training on Wikipedia enwik8 dataset
- PyTorch-based implementation
- Cosine similarity comparison with pretrained Gensim Word2Vec vectors
- Word analogy evaluation
- Basic bias detection in word embeddings

---

## Dataset

- **enwik8** (cleaned English Wikipedia text)
- Source: https://mattmahoney.net/dc/textdata.html

---

## Repository Structure

.
├── data/ # Dataset and preprocessed text
├── results/ # Saved embeddings and evaluation outputs
├── src/
│ ├── preprocess.py # Text preprocessing
│ ├── dataset.py # Vocabulary creation and negative sampling
│ ├── model.py # Skip-Gram model definition
│ ├── train.py # Training script
│ ├── evaluate_similarity.py
│ ├── analogy.py
│ └── bias.py
├── README.md
├── requirements.txt
└── Report.pdf

yaml
Copy code

---

## Setup and Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
Usage
1. Data Preprocessing
Prepare the Wikipedia text for training:

bash
Copy code
python src/preprocess.py
This generates a cleaned text file used for training.

2. Training
Train the Skip-Gram model:

bash
Copy code
python src/train.py
Default hyperparameters:

Embedding Dimension: 100

Window Size: 5

Negative Samples: 5

Epochs: 3

The trained embeddings are saved in the results/ directory.

3. Evaluation
Cosine Similarity Comparison (with Gensim pretrained vectors):

bash
Copy code
python src/evaluate_similarity.py
Word Analogy Task:

bash
Copy code
python src/analogy.py
Bias Detection:

bash
Copy code
python src/bias.py
Results
The learned embeddings capture meaningful semantic relationships between words.
Despite being trained on a relatively small corpus and limited epochs, the model demonstrates reasonable performance on similarity, analogy, and bias analysis tasks.

Notes
This implementation is intended for academic learning and experimentation.

Training is performed on a reduced dataset due to computational constraints.

References
Mikolov et al., “Efficient Estimation of Word Representations in Vector Space”, 2013

Jurafsky & Martin, Speech and Language Processing, Chapter 5

Author
Geda Jignash Babu



