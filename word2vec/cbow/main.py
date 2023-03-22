import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.autograd import Variable

import wandb
from model import CBOW
from raw_text import raw_text, raw_text1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script. '
                    'This enables running the different experiments while logging to a log-file and to wandb.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Arguments defining the training-process
    parser.add_argument('--batch_size', type=int, default=256, help=f'Batch size')
    parser.add_argument('--epochs', type=int, default=300, help=f'Number of epochs')
    parser.add_argument('--embedding_size', type=int, default=20, help=f'Dimension of embedding word')
    parser.add_argument('--context_size', type=int, default=2, help=f'Slided window of size context_size around the '
                                                                    f'target word.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help=f'Learning-rate of classifier')

    # Arguments for logging the training process.
    parser.add_argument('--path', type=str, default='./experiments', help=f'Output path for the experiment - '
                                                                          f'a sub-directory named with the data and '
                                                                          f'time will be created within')

    return parser.parse_args()


def data_preproccessing(context_size):
    vocabulary = set(raw_text)
    vocabulary_size = len(vocabulary)
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for i, word in enumerate(vocabulary)}

    data = list()

    for i in range(context_size, len(raw_text) - context_size):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    return data, vocabulary_size, word_to_idx, idx_to_word


def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


def train(model, epochs, data, word_to_idx, optimizer, loss_function):
    losses = list()
    logger.info('Start Training')
    num_iter = 0

    for epoch in range(epochs):
        total_loss = 0
        logger.info(f'epoch {epoch}')

        for context, target in data:
            wandb.log({"epoch": epoch}, step=num_iter)
            num_iter += 1

            context_vector = make_context_vector(context, word_to_idx)

            model.zero_grad()

            nll_prob = model(context_vector)
            loss = loss_function(nll_prob, Variable(torch.tensor([word_to_idx[target]])))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)

    logger.info(f'losses are {losses}')


def predict(word_to_idx, idx_to_word, model):
    context = ['process.', 'Computational', 'are', 'abstract']
    context_vector = make_context_vector(context, word_to_idx)

    a = model(context_vector).data.numpy()

    logger.info('Raw text: {}\n'.format(' '.join(raw_text)))
    logger.info('Test Context: {}\n'.format(context))

    max_idx = np.argmax(a)
    logger.info('Prediction: {}'.format(idx_to_word[max_idx]))


def model_pipeline(hyperparameters):
    # Tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        args = wandb.config

        logger.info(f'for {args.epochs} epochs '
                    f'bs={args.batch_size}, '
                    f'lr={args.learning_rate}, '
                    f'embedding_size={args.embedding_size}, '
                    f'context_size={args.context_size}, ')

        data, vocabulary_size, word_to_idx, idx_to_word = data_preproccessing(args.context_size)

        model = CBOW(args.embedding_size, vocabulary_size)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        loss_function = nn.NLLLoss()

        train(model, args.epochs, data, word_to_idx, optimizer, loss_function)
        predict(word_to_idx, idx_to_word, model)


@logger.catch
def main():
    args = parse_args()
    model_pipeline(hyperparameters=args)


if __name__ == '__main__':
    main()
