import unicodedata
import re
import nltk
from nltk.corpus import stopwords
import sklearn
from bs4 import BeautifulSoup

# nltk.download("popular")
# nltk.download('tagsets')

class Preprocessor():
    def __init__(self, texts):
        self.texts = texts
        self.corpus = ' '.join(self.texts)
        self.stop_words = set(stopwords.words('english'))
    
    def transform_to_lowercase(self, text=None):
        return text.lower()
    
    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text
        
    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
    
    def remove_special_characters(self, text):
        text = re.sub('[^a-zA-z0-9\s]', '', text)
        return text
        
#     def contaction(self, text):
#         return ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text(" ")])

    def clean(self, text=None, lower=True, strip_html=True, contract=True, remove_accented_chars=True,
              special_char_removal=True, remove_stop_words=True):
        if not text:
            text = self.corpus
        if lower:
            text = self.transform_to_lowercase(text)
        if strip_html:
            text = self.strip_html_tags(text)
        if remove_accented_chars:
            text = self.remove_accented_chars(text)
        if special_char_removal:
            text = self.remove_special_characters(text)
        if remove_stop_words:
            tokens = nltk.word_tokenize(text)
            cleaned_tokens = [word for word in tokens if word not in self.stop_words]
            text = ' '.join(cleaned_tokens)
        return text