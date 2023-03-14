from __future__ import print_function

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

lem = WordNetLemmatizer()


class SkipGramDataset(Dataset):
    def __init__(self, DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA):

        print("Parsing text and loading training data...")
        vocab, word_to_ix, ix_to_word, training_data = self.load_data(DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA)

        self.vocab = vocab
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word

        # training_data is a list of list of 2 indices
        self.data = torch.tensor(training_data, dtype=torch.long)

    def __getitem__(self, index):
        x = self.data[index, 0]
        y = self.data[index, 1]
        return x, y

    def __len__(self):
        return len(self.data)

    def create_trainig_data(self, split_text, word_to_ix, context_size):
        training_data = list()

        print('preparing training data (x, y)...')
        for sentence in tqdm(split_text):
            indices = [word_to_ix[word] for word in sentence]

            # for each word treated as center word
            for center_word_pos in range(len(indices)):

                # for each window  position
                for word_context_idx in range(-context_size, context_size + 1):
                    context_word_pos = center_word_pos + word_context_idx

                    # make sure we dont jump out of the sentence
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue

                    context_word_idx = indices[context_word_pos]
                    center_word_idx = indices[center_word_pos]

                    training_data.append([center_word_idx, context_word_idx])

        return training_data

    def gather_word_freqs(self, split_text):

        vocab = {}
        ix_to_word = {}
        word_to_ix = {}
        total = 0.0

        print('building vocab...')
        for word_tokens in tqdm(split_text):
            for word in word_tokens:  # for every word in the word list(split_text), which might occur multiple times
                if word not in vocab:  # only new words allowed
                    vocab[word] = 0

                    # Assign new id for new word
                    ix_to_word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)

                vocab[word] += 1.0  # count of the word stored in a dict
                total += 1.0  # total number of words in the word_list(split_text)

        return split_text, vocab, word_to_ix, ix_to_word

    def load_data(self, data_source, context_size, fraction_data):

        stop_words = set(stopwords.words('english'))

        if data_source == 'toy':
            sentences = [
                'word1 word2 word3 word4 word5',
                'word6 word7 word8 word9 word10',
                'word11 word12 word13 word14 word15'
            ]
            # sents = ['word6 word7 word8 word9 word10', 'word1 word1 word1 word2 word2 word3 word4 word5', 'word11 word12 word13 word14 word15']

        elif data_source == 'gensim':
            import gensim.downloader as api
            dataset = api.load("text8")
            data = [d for d in dataset][:int(fraction_data * len([d_ for d_ in dataset]))]
            print(f'fraction of data taken: {fraction_data}/1')

            sentences = []
            print('forming sentences by joining tokenized words...')
            for d in tqdm(data):
                sentences.append(' '.join(d))

        sentences_list_tokenized = [word_tokenize(sentence) for sentence in sentences]
        print('len(sent_list_tokenized): ', len(sentences_list_tokenized))

        sent_list_tokenized_filtered = []
        print('lemmatizing and removing stopwords...')
        for sentence in tqdm(sentences_list_tokenized):
            # Lemmatize is text normalization technique, that switches any kind of a word to its base root mode.
            sent_list_tokenized_filtered.append([lem.lemmatize(word, 'v') for word in sentence if word not in stop_words])

        sent_list_tokenized_filtered, vocab, word_to_ix, ix_to_word = self.gather_word_freqs(sent_list_tokenized_filtered)

        training_data = self.create_trainig_data(sent_list_tokenized_filtered, word_to_ix, context_size)

        return vocab, word_to_ix, ix_to_word, training_data
