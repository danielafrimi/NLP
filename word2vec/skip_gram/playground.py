from skip_gram.dataset import SkipGramDataset

DATA_SOURCE = 'gensim' # or 'toy'
CONTEXT_SIZE = 3
FRACTION_DATA = 0.01

s = SkipGramDataset(DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA)




