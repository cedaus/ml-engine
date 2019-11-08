from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from vector import Vectorizer
from data_explorer import DataExplorer

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D, LSTM, Bidirectional, InputLayer, SimpleRNN

class Model():
    def __init__(self, train_texts, train_labels, test_texts, test_labels, word_embedding=False):
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.vocab = None
        self.vectorizer = None
        self.train_vector = None
        self.test_vector = None
        self.prediction = None
        self.max_features = 20000
        self.ngram_range = (1,2)
        self.word_embedding = word_embedding
        
    def vectorize(self):
        V = Vectorizer(train_texts=self.train_texts, test_texts=self.test_texts, max_features=self.max_features)
        if not self.word_embedding:
            self.train_vector, self.test_vector, self.vocab = V.tfidf_vectorize(
                {'strip_accents': 'unicode',
                 'analyzer': 'word',
                 'ngram_range': self.ngram_range,
                 'min_df': 2,
                 'max_features': self.max_features
                })
        else:
            # code here
            return
        
    def run(self, classifier):
        self.vectorize()
        model = classifier().fit(self.train_vector, self.train_labels)
        self.prediction = model.predict(self.test_vector)
        print(confusion_matrix(self.test_labels, self.prediction))  
        print(classification_report(self.test_labels, self.prediction))  
        print(accuracy_score(self.test_labels, self.prediction))
        

class NNModel():
    def __init__(self, num_classes, train_texts, train_labels, test_texts, test_labels):
        self.num_classes = num_classes
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.train_vector = None
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.test_vector = None
        self.vocab = None
        #
        self.MAX_FEATURES = 20000
        self.MAX_SEQUENCE_LENGTH = 500
        #
        DE = DataExplorer(self.train_texts)
        self.num_texts, self.total_words, self.avg_words_text = DE.get_corpus_statistics()
        self.S_by_W = self.num_texts / self.avg_words_text
        # Layer's Params
        self.input_shape = None
        self.KERNAL_SIZE = 3
        self.DROPOUT_RATE = 0.2
        self.UNITS = 64
        self.LAST_LAYER_UNITS = None
        self.LAST_LAYER_ACTIVATION = None
        # Embedding
        self.EMBEDDING_DIM = 300
        self.EMBEDDING_WEIGHTS = None
        self.word_vect_dic = None
        # Convolution
        self.POOL_SIZE = None
        self.filters = None
        #
        self.OPTIMIZER = 'adam'
        self.METRIC = 'accuracy'
        self.LOSS = None
        self.LEARNING_RATE = 1e-3
        self.EPOCHS = 500
        self.BATCH_SIZE = 128
        
    
    def set_params(self, input_shape=None, filters=None, units=None,
                   kernal_size=None, pool_size=None, dropout_rate=None,
                   learning_rate=None, epochs=None, batch_size=None, embed_dim=None):        
        if input_shape:
            self.input_shape = input_shape
        if filters:
            self.filters = filters
        if units:
            self.UNITS = units
        if kernal_size:
            self.kernal_size = kernal_size
        if pool_size:
            self.pool_size = pool_size
        if dropout_rate:
            self.dropout_rate = dropout_rate
        if learning_rate:
            self.learning_rate = learning_rate
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if embed_dim:
            self.embed_dim = embed_dim
            
        if self.num_classes == 2:
            self.last_layer_activation = 'sigmoid'
            self.last_layer_units = 1
            self.loss = 'binary_crossentropy'
        elif self.num_classes > 2:
            self.last_layer_activation = 'softmax'
            self.last_layer_units = self.num_classes
            self.loss = 'sparse_categorical_crossentropy'
        else:
            print('ERROR')
            
    def vectorize(self):
        V = Vectorizer(train_texts=self.train_texts, test_texts=self.test_texts, max_features=self.max_features)
        if self.S_by_W < 1500:
            self.train_vector, self.test_vector, self.vocab = V.tfidf_vectorize(
                {'strip_accents': 'unicode',
                 'analyzer': 'word',
                 'ngram_range': (1, 2),
                 'min_df': 2,
                 'max_features': self.max_features
                })
        else:
            self.train_vector, self.test_vector, self.vocab = V.sequence_vectorize()
            self.embedding_matrix, self.word_vect_dic = V.word_embedding_vectorize(self.vocab)
        
        
    def get_Embedding(self, use_pretrained_embedding=False, is_embedding_trainable=False):
        if use_pretrained_embedding:
            layer = Embedding(
                input_dim=len(self.vocab) + 1,
                output_dim=self.EMBEDDING_DIM,
                input_length=self.MAX_SEQUENCE_LENGTH,
                weights=[self.embedding_matrix],
                trainable=is_embedding_trainable
               )
        else:
            layer = Embedding(
                input_dim=num_features,
                output_dim=self.EMBEDDING_DIM,
                input_length=self.input_shape[0]
            )
         
        return layer
    
    def get_SeparableConv1D(n):
        layer = SeparableConv1D(
            filters=self.filters * n,
            kernel_size=self.kernel_size,
            activation='relu',
            bias_initializer='random_uniform',
            depthwise_initializer='random_uniform',
            padding='same'
        )
        return layer
    
    def get_Conv1D(n):
        layer = Conv1D(
            filters=self.filters * n,
            kernel_size=self.kernel_size,
            activation='relu',
        )
        return layer
    
    def build_mlp_model(self, layers):
        """
        Multi Layer Perceptrons (MLPs)
        """
        model = models.Sequential()
        model.add(Dropout(rate=self.dropout_rate, input_shape=self.input_shape))
        
        for _ in range(layers-1):
            model.add(Dense(units=self.UNITS, activation='relu'))
            model.add(Dropout(rate=self.DROPOUT_RATE))
        
        model.add(Dense(units=self.LAST_LAYER_UNITS, activation=self.LAST_LAYER_ACTIVATION))
        return model
    
    def build_cnn_model(self, layers, n):
        """
        Convolutional Neural Network
        """
        model = models.Sequential()
        model.add(self.get_Embedding())
        
        for _ in range(layers - 1):
            model.add(self.get_Conv1D(n))
            model.add(MaxPooling1D(pool_size=self.POOL_SIZE))
            
        model.add(Dense(units=self.UNITS, ativation='relu'))
        model.add(Dense(units=self.LAST_LAYER_UNITS, activation=self.LAST_LAYER_ACTIVATION))
        
        return model
    
    def build_sepcnn_model(self, blocks):
        """
        Separable Convolutional Network
        """
        model = models.Sequential()
        model.add(self.get_Embedding())
        
        for _ in range(blocks - 1):
            model.add(Dropout(rate=self.dropout_rate))
            model.add(self.get_SeparableConv1D(1))
            model.add(self.get_SeparableConv1D(1))
            model.add(MaxPooling1D(pool_size=pool_size))
            
        model.add(self.get_SeparableConv1D(2))
        model.add(self.get_SeparableConv1D(2))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(rate=self.DROPOUT_RATE))
        model.add(Dense(units=self.LAST_LAYER_UNITS, activation=self.LAST_LAYER_ACTIVATION))
            
        return model
    
    def build_rnn_model(self):
        model = models.Sequential()
        model.add(InputLayer(input_shape=(15,1)))
        model.add(self.get_Embedding())
        
        model.add(SimpleRNN(units = 100, activation='relu', use_bias=True))
        model.add(Dense(units=1000, input_dim = 2000, activation='sigmoid'))
        model.add(Dense(units=500, input_dim=1000, activation='relu'))
        model.add(Dense(units=self.LAST_LAYER_UNITS, input_dim=500, activation=self.LAST_LAYER_ACTIVATION))
        
        return model
    
    def build_UniLSTM(self):
        """
        Unidirectional LSTM Model
        """
        model = models.Sequential()
        model.add(self.get_Embedding())
        model.add(LSTM(self.UNITS))
        model.add(Dense(units=self.UNITS, activation='relu'))
        model.add(Dropout(rate=self.DROPOUT_RATE))
        model.add(Dense(units=self.LAST_LAYER_UNITS, activation=self.LAST_LAYER_ACTIVATION))
        
        return model
    
    def build_BiLSTM(self):
        """
        Bidirectional LSTM Model
        """
        model = models.Sequential()
        model.add(self.get_Embedding())
        model.add(Bidirectional(LSTM(self.UNITS)))
        model.add(Dense(units=self.UNITS, activation='relu'))
        model.add(Dropout(rate=self.DROPOUT_RATE))
        model.add(Dense(units=self.LAST_LAYER_UNITS, activation=self.LAST_LAYER_ACTIVATION))
        
        return model
    
    def build_han_model(self):
        """
        Incomplete and wrong
        """
        model = models.Sequential()
        model.add(Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32'))
        model.add(self.get_Embedding())
        model.add(Bidirectional(LSTM(self.UNITS, return_sequences=True)))
        model.add(TimeDistributed(Dense(200)))
        model.add(AttentionWithContext())
        
    
    def run(self):
        self.set_params()
        self.vectorize()
        
        if self.S_by_W < 1500:
            """
            N-gram Model
            """
            self.input_shape = self.train_vector.shape[1:]
            model = self.build_mlp_model(layers=2)
            model.compile(optimizer=self.OPTIMIZER, loss=self.loss, metrics=[self.metric])
            print(model)
        
            history = model.fit(
            self.train_vector,
            self.train_labels,
            epochs=self.epochs,
            validation_data=(self.test_vector, self.test_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=self.batch_size).history
        
            print('Validation accuracy: {acc}, loss: {loss}'.format(
                acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
        else:
            """
            Sequence Model
            """
            self.build_sepcnn_model()