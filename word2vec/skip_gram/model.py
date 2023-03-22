from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, embedding_size, vocab_size, noise_dist=None, negative_samples=10):
        super(SkipGram, self).__init__()

        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.noise_dist = noise_dist

        # Initialize both embedding tables with uniform distribution
        self.embeddings_input.weight.data.uniform_(-1, 1)
        self.embeddings_context.weight.data.uniform_(-1, 1)

    def forward(self, input_word, context_word):
        # computing out loss
        emb_input = self.embeddings_input(input_word)
        emb_context = self.embeddings_context(context_word)
        emb_product = torch.mul(emb_input, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out_loss = F.logsigmoid(emb_product)

        if self.negative_samples > 0:
            # computing negative loss
            noise_dist = self.noise_dist

            # Create negative examples for the context word
            num_neg_samples_for_this_batch = context_word.shape[0] * self.negative_samples

            # Returns a tensor where each row contains num_samples indices sampled from the multinomial probability
            # distribution located in the corresponding row of tensor input.
            negative_example = torch.multinomial(noise_dist, num_neg_samples_for_this_batch,
                                                 replacement=True)  # coz bs*num_neg_samples > vocab_size

            negative_example = negative_example.view(context_word.shape[0], self.negative_samples) # bs, num_neg_samples
            emb_negative = self.embeddings_context(negative_example)  # bs, neg_samples, emb_dim
            emb_product_neg_samples = torch.bmm(emb_negative.neg(), emb_input.unsqueeze(2))  # bs, neg_samples, 1
            noise_loss = F.logsigmoid(emb_product_neg_samples).squeeze(2).sum(1)  # bs
            total_loss = -(out_loss + noise_loss).mean()

            return total_loss

        else:
            return -(out_loss).mean()
