import torch
import torch.optim as optim
import numpy as np
import random
from dataset import SkipGramDataset
from model import SkipGramNS
from tqdm import tqdm
import pickle
import os

# ---------------- CONFIG ----------------
EMBEDDING_DIM = 100
WINDOW_SIZE = 5
MIN_COUNT = 5
NEG_SAMPLES = 5
BATCH_SIZE = 1024
EPOCHS = 3
LR = 0.001

DATA_PATH = "data/wiki_clean.txt"
SAVE_DIR = "results"
# ---------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset
dataset = SkipGramDataset(
    file_path=DATA_PATH,
    window_size=WINDOW_SIZE,
    min_count=MIN_COUNT
)

vocab_size = len(dataset.idx2word)
print("Vocab size:", vocab_size)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGramNS(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Negative sampling distribution
neg_dist = torch.tensor(dataset.neg_sampling_dist)

# Helper: generate batch
def generate_batch(batch_size):
    centers = []
    contexts = []

    while len(centers) < batch_size:
        idx = random.randint(0, len(dataset.data) - 1)
        center = dataset.data[idx]

        window = random.randint(1, WINDOW_SIZE)
        context_idx = idx + random.choice(
            list(range(-window, 0)) + list(range(1, window + 1))
        )

        if context_idx < 0 or context_idx >= len(dataset.data):
            continue

        centers.append(center)
        contexts.append(dataset.data[context_idx])

    return centers, contexts


# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    total_loss = 0
    steps = 10000  # number of updates per epoch

    for _ in tqdm(range(steps), desc=f"Epoch {epoch+1}"):
        center_words, pos_words = generate_batch(BATCH_SIZE)
        # ---- FIX 2: Safety check ----
        assert len(center_words) == BATCH_SIZE
        assert len(pos_words) == BATCH_SIZE
        # ----------------------------
        center_words = torch.tensor(center_words).to(device)
        pos_words = torch.tensor(pos_words).to(device)

        neg_words = torch.multinomial(
            neg_dist,
            BATCH_SIZE * NEG_SAMPLES,
            replacement=True
        ).view(BATCH_SIZE, NEG_SAMPLES).to(device)

        loss = model(center_words, pos_words, neg_words)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / steps
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# ---------------- SAVE ----------------
torch.save(model.in_embeddings.weight.data.cpu(), f"{SAVE_DIR}/embeddings.pt")

with open(f"{SAVE_DIR}/word2idx.pkl", "wb") as f:
    pickle.dump(dataset.word2idx, f)

with open(f"{SAVE_DIR}/idx2word.pkl", "wb") as f:
    pickle.dump(dataset.idx2word, f)

print("Training complete. Embeddings saved.")
