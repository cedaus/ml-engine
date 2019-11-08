import numpy as np
import matplotlib.pyplot as plt
import nltk

from preprocessor import Preprocessor

class DataExplorer():
    def __init__(self, texts):
        self.texts = texts
        self.corpus = ' '.join(self.texts)
        
    def get_num_words_per_sample(self):
        numWords = []
        for text in self.texts:
            counter = len(text.split())
            numWords.append(counter)  
            
        return numWords
        
    def get_median_num_words(self):
        """Returns the median number of words per sample given corpus.

        # Arguments
            sample_texts: list, sample texts.

        # Returns
            int, median number of words per sample.
        """
        num_words = [len(s.split()) for s in self.texts]
        return np.median(num_words)
    
    def plot_sample_length_distribution(self):
        """Plots the sample length distribution.

        # Arguments
            samples_texts: list, sample texts.
        """
        plt.hist([len(s) for s in self.texts], 50)
        plt.xlabel('Length of a sample')
        plt.ylabel('Number of samples')
        plt.title('Sample length distribution')
        plt.show()
        
    def plot_frequency_distribution_of_ngram(self):
        return None
    
    def plot_most_frequent_words(self):
        # Visualization of the most frequent words
        words = nltk.word_tokenize(self.corpus)
        fdist = nltk.FreqDist(words)
        print('Number of tokens:', len(words))
        print("List of 100 most frequent words/counts")
        print(fdist.most_common(100))
        fdist.plot(40)
        
    def plot_most_frequent_words_preprocessed(self):
        P = Preprocessor(self.texts)
        prep_corpus = P.clean()
        words = nltk.word_tokenize(prep_corpus)
        fdist = nltk.FreqDist(words)
        print('Number of tokens:', len(words))
        print("List of 100 most frequent words/counts")
        print(fdist.most_common(100))
        fdist.plot(40)
        
    def get_corpus_statistics(self):
        # Retrieve some info on the text data
        num_texts = len(self.texts)
        total_words = len(nltk.word_tokenize(self.corpus))
        avg_words_text = self.get_median_num_words()
        
        print('Number of texts:', num_texts)
        print('The total number of words in all texts', total_words)
        print('The average number of words in each text is', avg_words_text)
              
        return num_texts, total_words, avg_words_text