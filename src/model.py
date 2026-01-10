import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # Input embeddings (center words)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Output embeddings (context words)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self._init_embeddings()

    def _init_embeddings(self):
        # Initialize embeddings uniformly
        init_range = 0.5 / self.in_embeddings.embedding_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, center_words, pos_context_words, neg_context_words):
        """
        center_words:        (batch_size)
        pos_context_words:   (batch_size)
        neg_context_words:   (batch_size, num_negatives)
        """

        # Get embeddings
        center_embeds = self.in_embeddings(center_words)            # (B, D)
        pos_embeds = self.out_embeddings(pos_context_words)         # (B, D)
        neg_embeds = self.out_embeddings(neg_context_words)         # (B, K, D)

        # Positive score
        pos_score = torch.sum(center_embeds * pos_embeds, dim=1)   # (B)
        pos_loss = F.logsigmoid(pos_score)

        # Negative score
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        # Final loss
        loss = -(pos_loss + neg_loss).mean()

        return loss
