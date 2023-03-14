from __future__ import print_function

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from skip_gram.Constants import *
from skip_gram.dataset import SkipGramDataset
from skip_gram.model import SkipGram

train_dataset = SkipGramDataset(DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA)

vocab = train_dataset.vocab
word_to_ix = train_dataset.word_to_ix
ix_to_word = train_dataset.ix_to_word

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
print('len(train_dataset): ', len(train_dataset))
print('len(train_loader): ', len(train_loader))
print('len(vocab): ', len(vocab), '\n')

# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs / sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

model = SkipGram(EMBEDDING_DIM, len(vocab), noise_dist, NEGATIVE_SAMPLES)

optimizer = optim.Adam(model.parameters(), lr=LR)

losses = list()
for epoch in tqdm(range(NUM_EPOCHS)):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):

        model.train()

        optimizer.zero_grad()
        loss = model(x_batch, y_batch)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(loss.item())


        # if batch_idx % DISPLAY_EVERY_N_BATCH == 0 and DISPLAY_BATCH_LOSS:
        #     print(f'Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}')
            # show 5 closest words to some test words
            # print_nearest_words(model, TEST_WORDS, word_to_ix, ix_to_word, top=5)

    # write embeddings every SAVE_EVERY_N_EPOCH epoch
    # if epoch % SAVE_EVERY_N_EPOCH == 0:
    #     writer.add_embedding(model.embeddings_input.weight.data,
    #                          metadata=[ix_to_word[k] for k in range(len(ix_to_word))], global_step=epoch)
    #
    #     torch.save({'model_state_dict': model.state_dict(),
    #                 'losses': losses,
    #                 'word_to_ix': word_to_ix,
    #                 'ix_to_word': ix_to_word
    #                 },
    #                '{}/model{}.pth'.format(MODEL_DIR, epoch))

# plt.figure(figsize=(50, 50))
# plt.xlabel("batches")
# plt.ylabel("batch_loss")
# plt.title("loss vs #batch")
#
# plt.plot(losses)
# plt.savefig('losses.png')
# plt.show()
#
# # '''
# EMBEDDINGS = model.embeddings_input.weight.data
# print('EMBEDDINGS.shape: ', EMBEDDINGS.shape)
#
# from sklearn.manifold import TSNE
#
# print('\n', 'running TSNE...')
# tsne = TSNE(n_components=2).fit_transform(EMBEDDINGS.cpu())
# print('tsne.shape: ', tsne.shape)  # (15, 2)
#
# ############ VISUALIZING ############
# x, y = [], []
# annotations = []
# for idx, coord in enumerate(tsne):
#     # print(coord)
#     annotations.append(ix_to_word[idx])
#     x.append(coord[0])
#     y.append(coord[1])
#
# # test_words = ['king', 'queen', 'berlin', 'capital', 'germany', 'palace', 'stays']
# # test_words = ['sun', 'moon', 'earth', 'while', 'open', 'run', 'distance', 'energy', 'coal', 'exploit']
# # test_words = ['amazing', 'beautiful', 'work', 'breakfast', 'husband', 'hotel', 'quick', 'cockroach']
#
# test_words = TEST_WORDS_VIZ
# print('test_words: ', test_words)
#
# plt.figure(figsize=(50, 50))
# for i in range(len(test_words)):
#     word = test_words[i]
#     # print('word: ', word)
#     vocab_idx = word_to_ix[word]
#     # print('vocab_idx: ', vocab_idx)
#     plt.scatter(x[vocab_idx], y[vocab_idx])
#     plt.annotate(word, xy=(x[vocab_idx], y[vocab_idx]), \
#                  ha='right', va='bottom')
#
# plt.savefig("w2v.png")
# plt.show()
# # '''