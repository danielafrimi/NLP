from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.manifold import TSNE

from skip_gram.dataset import SkipGramDataset
from skip_gram.model import SkipGram


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script. '
                    'This enables running the different experiments while logging to a log-file and to wandb.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Arguments defining the training-process
    parser.add_argument('--batch_size', type=int, default=256, help=f'Batch size')
    parser.add_argument('--data_source', type=str, default='gensim', help=f'Data Source to train the model')
    parser.add_argument('--data_fraction', type=float, default=0.2, help=f'Fraction of the data to train for')
    parser.add_argument('--epochs', type=int, default=300, help=f'Number of epochs')
    parser.add_argument('--num_negative_samples', type=int, default=3, help=f'Number of negative samples to train in '
                                                                            f'each forward pass of the net')
    parser.add_argument('--embedding_size', type=int, default=100, help=f'Dimension of embedding word')
    parser.add_argument('--context_size', type=int, default=2, help=f'Slided window of size context_size around the '
                                                                    f'target word.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help=f'Learning-rate of classifier')

    # Arguments for logging the training process.
    parser.add_argument('--path', type=str, default='./experiments', help=f'Output path for the experiment - '
                                                                          f'a sub-directory named with the data and '
                                                                          f'time will be created within')

    return parser.parse_args()


def train_model(model, train_loader, learning_rate, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = list()
    for epoch in tqdm(range(num_epochs)):
        logger.info('\n===== EPOCH {}/{} ====='.format(epoch + 1, num_epochs))

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            model.train()

            optimizer.zero_grad()
            loss = model(x_batch, y_batch)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            logger.info(loss.item())


def visuzilze_embedded(model, ix_to_word, word_to_ix):
    # Take the embeddings_input of the model after train completed.
    # Those latent vector are with semantic meaning.
    embeddings_input_layer = model.embeddings_input.weight.data
    logger.info('EMBEDDINGS.shape: ', embeddings_input_layer.shape)

    tsne = TSNE(n_components=2).fit_transform(embeddings_input_layer.cpu())

    x, y = [], []
    annotations = []
    for idx, coord in enumerate(tsne):
        # print(coord)
        annotations.append(ix_to_word[idx])
        x.append(coord[0])
        y.append(coord[1])

    test_words = ['human', 'boy', 'office', 'woman']

    plt.figure(figsize=(50, 50))
    for i in range(len(test_words)):
        word = test_words[i]
        vocab_idx = word_to_ix[word]

        plt.scatter(x[vocab_idx], y[vocab_idx])
        plt.annotate(word, xy=(x[vocab_idx], y[vocab_idx]), ha='right', va='bottom')

    plt.savefig("w2v.png")
    plt.show()


def model_pipeline(hyperparameters):
    train_dataset = SkipGramDataset(hyperparameters.data_source, hyperparameters.context_size,
                                    hyperparameters.data_fraction)

    vocab = train_dataset.vocab
    word_to_ix = train_dataset.word_to_ix
    ix_to_word = train_dataset.ix_to_word

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters.batch_size, shuffle=False)

    # make noise distribution to sample negative examples from
    word_freqs = np.array(list(vocab.values()))
    unigram_dist = word_freqs / sum(word_freqs)
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

    model = SkipGram(hyperparameters.embedding_size, len(vocab),
                     noise_dist, hyperparameters.num_negative_samples)

    train_model(model, train_loader, hyperparameters.learning_rate, hyperparameters.epochs)
    visuzilze_embedded(model, ix_to_word, word_to_ix)


@logger.catch
def main():
    args = parse_args()
    model_pipeline(hyperparameters=args)


if __name__ == '__main__':
    main()
