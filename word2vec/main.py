import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import CBOW

import torch.optim as optim
import wandb
from loguru import logger


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
    parser.add_argument('--learning_rate_classifiers', type=float, default=0.001, help=f'Learning-rate of classifier')

    # Arguments for logging the training process.
    parser.add_argument('--path', type=str, default='./experiments', help=f'Output path for the experiment - '
                                                                          f'a sub-directory named with the data and '
                                                                          f'time will be created within')

    return parser.parse_args()


args = parse_args()


raw_text = """Zebras are African equines with 
distinctive black-and-white striped coats. There are three living species: the Grévy's zebra (Equus grevyi), 
plains zebra, and the mountain zebra. Zebras share the genus Equus with horses and asses, 
the three groups being the only living members of the family Equidae. Zebra stripes come in different patterns, 
unique to each individual. Several theories have been proposed for the function of these stripes, with most evidence 
supporting them as a deterrent for biting flies. Zebras inhabit eastern and southern Africa and can be found in a 
variety of habitats such as savannahs, grasslands, woodlands, shrublands, and mountainous areas. 

Zebras are primarily grazers and can subsist on lower-quality vegetation. They are preyed on mainly by lions, 
and typically flee when threatened but also bite and kick. Zebra species differ in social behaviour, with plains and 
mountain zebra living in stable harems consisting of an adult male or stallion, several adult females or mares, 
and their young or foals; while Grévy's zebra live alone or in loosely associated herds. In harem-holding species, 
adult females mate only with their harem stallion, while male Grévy's zebras establish territories which attract 
females and the species is promiscuous. Zebras communicate with various vocalisations, body postures and facial 
expressions. Social grooming strengthens social bonds in plains and mountain zebras. 

Zebras' dazzling stripes make them among the most recognisable mammals. They have been featured in art and stories in 
Africa and beyond. Historically, they have been highly sought after by exotic animal collectors, but unlike horses 
and donkeys, zebras have never been truly domesticated. The International Union for Conservation of Nature (IUCN) 
lists the Grévy's zebra as endangered, the mountain zebra as vulnerable and the plains zebra as near-threatened. The 
quagga, a type of plains zebra, was driven to extinction in the 19th century. Nevertheless, 
zebras can be found in numerous protected areas. """

raw_text2 = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()


def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


vocabulary = set(raw_text)
vocabulary_size = len(vocabulary)

word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

data = []

for i in range(args.context_size, len(raw_text) - args.context_size):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

model = CBOW(args.embedding_size, vocabulary_size)
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = []
loss_function = nn.NLLLoss()

for epoch in range(100):
    total_loss = 0
    print(f'epoch {epoch}')
    for context, target in data:
        context_vector = make_context_vector(context, word_to_idx)

        # Remember PyTorch accumulates gradients; zero them out
        model.zero_grad()

        nll_prob = model(context_vector)
        loss = loss_function(nll_prob, Variable(torch.tensor([word_to_idx[target]])))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)

print(losses)

# Let's see if our CBOW model works or not

print("*************************************************************************")

context = ['process.', 'Computational', 'are', 'abstract']
context_vector = make_context_vector(context, word_to_idx)
a = model(context_vector).data.numpy()
print('Raw text: {}\n'.format(' '.join(raw_text)))
print('Test Context: {}\n'.format(context))
max_idx = np.argmax(a)
print('Prediction: {}'.format(idx_to_word[max_idx]))

if __name__ == '__main__':
    pass
