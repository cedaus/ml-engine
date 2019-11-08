import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors
from gensim.models import word2vec

class Vectorizer():
    def __init__(self, train_texts, test_texts, max_features=20000, max_sequence_length=500):
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.preprocessed_train_corpus = Preprocessor(self.train_texts).clean()
        self.preprocessed_test_corpus = Preprocessor(self.test_texts).clean()
        self.max_features = max_features
        self.max_sequence_length = max_sequence_length
        self.embed_dim = 300

    def get_custom_params(self, ngram_range=None, stop_words=None, min_df=None, max_dif=None, tokenizer=None,
                          analyzer=None, preprocessor=None, lowercase=None, max_features=None, dtype=None, strip_accents=None):
        params = {}
        if ngram_range:
            params['ngram_range'] = ngram_range
        if stop_words:
            params['stop_words'] = stop_words
        if min_df:
            params['min_df'] = min_df
        if max_df:
            params['max_df'] = max_df
        if tokenizer:
            params['tokenizer'] = tokenizer
        if analyzer:
            params['analyzer'] = analyzer
        if preprocessor:
            params['preprocessor'] = preprocessor
        if lowercase:
            params['lowercase'] = lowercase
        if max_features:
            params['lowercase'] = max_features
        if dtype:
            params['dtype'] = dtype
        if strip_accents:
            params['strip_accents'] = strip_accents

        return params

    def get_vector_info(vector):
        matrix = vector.toarray()
        shape = vector.shape
        return matrix, shape

    def count_vectorize(self, kwargs):
        vectorizer = CountVectorizer(**kwargs)
        train_vector = vectorizer.fit_transform(self.train_texts)
        test_vector = vectorizer.transform(self.test_texts)
        # List of features (Words)
        features = vectorizer.get_feature_names()
        # Index assigned for every token
        vocabulary = vectorizer.vocabulary_
        return train_vector, test_vector, vocabulary

    def tfidf_vectorize(self, kwargs):
        vectorizer = TfidfVectorizer(**kwargs)
        train_vector = vectorizer.fit_transform(self.train_texts)
        test_vector = vectorizer.transform(self.test_texts)
        # List of features (Words)
        features = vectorizer.get_feature_names()
        # Index assigned for every token
        vocabulary = vectorizer.vocabulary_
        return train_vector, test_vector, vocabulary

    def sequence_vectorize(self):
        # Create vocabulary with training texts.
        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(pd.Series(self.preprocessed_train_corpus))

        # Vectorize texts
        train_vector = tokenizer.texts_to_sequences(pd.Series(self.preprocessed_train_corpus))
        test_vector = tokenizer.texts_to_sequences(pd.Series(self.preprocessed_test_corpus))

        # Get max sequence length.
        max_length = len(max(train_vector, key=len))
        if max_length > self.max_sequence_length:
            max_length = self.max_sequence_length

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated at the beginning.
        train_vector = pad_sequences(train_vector, maxlen=max_length)
        test_vector = pad_sequences(test_vector, maxlen=max_length)
        # Index assigned for every token
        vocabulary = tokenizer.word_index

        return train_vector, test_vector, vocabulary

    def word2vec_vectorize(self, vocab):
        word2vec = KeyedVectors.load_word2vec_format('Data/GoogleNews-vectors.bin.gz',binary=True)

        # Construct the embedding weights matrix
        # Where rows is length of vocab + 1
        # And column is value of embed_dim
        embedding_weights = np.zeros((len(vocab) + 1, self.embed_dim))
        # Creating a dictionary item of vocab
        for word, index in vocab.items():
            embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(self.embed_dim)

        # Constructing word-vector dictionary
        word_vector_dict = dict(zip(pd.Series(list(vocab.keys())),
                                    pd.Series(list(vocab.keys())).apply(
                                        lambda x: features_embedding_weights[vocab[x]]
                                    )))

        return embedding_weights, word_vector_dict

    def tfidf_embedding_vectorize(self):
        return
