

### Text Clustering using NLP techniques <br>
In recent years, Natural Language Processing (NLP) has become increasingly popular as a tool for analyzing large volumes of text data.

However, with so much information available, it can be difficult to make sense of it all. This is where text clustering comes in.

Text clustering is the process of grouping similar documents together based on their content. By clustering text, 
we can identify patterns and trends that would otherwise be difficult to discern.

This technique has many applications, from market research to customer segmentation to sentiment analysis. In this blog post, 
we will explore how text clustering can be used to analyze text data and uncover insights that can be used to make better business decisions.


The notebook focused on text clustering using various embedding techniques. The dataset we are using is the 20newsgroups dataset with 3 categories. 
The goal is to compare several embedding approaches such as sentence transformers, GloVe, TF-IDF, and BERT-CLS, and cluster the resulting embeddings. 
This comparison can help to determine which approach provides the best clustering performance for the given dataset.

There are several embedding techniques available, each with its own strengths and weaknesses.

### Techniques

#### TF-IDF Vectorization
This is a simple but effective method for generating vector representations of sentences. 
It stands for “term frequency-inverse document frequency” and it calculates the importance of words in a sentence by taking 
into account how often they appear in the sentence and how rare they are in the entire corpus of sentences.

#### Sentence Transformer
Sentence Transformers are deep learning models that can encode natural language sentences into high-dimensional vector representations. They are trained using a pre-training and fine-tuning approach and have achieved state-of-the-art performance on several natural language processing tasks. These models are widely used for various applications such as chatbots, search engines, and recommendation systems.

#### Glove
GloVe is a word embedding technique that represents words as dense vectors in a high-dimensional space. 
It captures both local and global context, making it useful for various tasks. To cluster sentences using GloVe, one approach is to concatenate the word vectors in a sentence, form a matrix, and then apply a clustering algorithm such as k-means. The resulting clusters can reveal common themes or patterns in the data.

#### BERT — [CLS] token for sentence context
BERT, is a pre-trained deep learning model that can be fine-tuned for various natural language processing tasks. One of the main innovations of BERT is its ability to represent both the left and right context of a word, 
allowing it to better capture the meaning of a sentence.

In BERT, the [CLS] token, which stands for “classification”, is a special token that is inserted at the beginning of every input sequence. During pre-training, BERT is trained to predict the correct class label for the entire sequence based on the [CLS] token representation,
which is meant to capture the overall meaning of the sequence.



Read about this in my medium blog https://medium.com/@danielafrimi/text-clustering-using-nlp-techniques-c2e6b08b6e95
