from collections import Counter
import numpy as np

class SkipGramDataset:
    def __init__(self, file_path, window_size=5, min_count=5):
        self.window_size = window_size

        # Load text
        with open(file_path, "r", encoding="utf-8") as f:
            words = f.read().split()

        print("Total tokens:", len(words))

        # Build vocabulary
        counts = Counter(words)
        self.word2idx = {}
        self.idx2word = []

        for word, freq in counts.items():
            if freq >= min_count:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

        print("Vocabulary size:", len(self.idx2word))

        # Convert words to indices
        self.data = [self.word2idx[w] for w in words if w in self.word2idx]

        # Unigram distribution for negative sampling
        freqs = np.array([counts[w] for w in self.idx2word], dtype=np.float32)
        self.neg_sampling_dist = freqs ** 0.75
        self.neg_sampling_dist /= self.neg_sampling_dist.sum()

    def generate_pairs(self):
        pairs = []
        for i, center in enumerate(self.data):
            start = max(0, i - self.window_size)
            end = min(len(self.data), i + self.window_size + 1)

            for j in range(start, end):
                if i != j:
                    pairs.append((center, self.data[j]))

        return pairs
