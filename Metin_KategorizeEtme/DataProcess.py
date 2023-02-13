import re
import string
from nltk.tokenize import word_tokenize
class DataProcess():
    def tokenizasyon(self,text):
        return word_tokenize(text)

    def convert_lowercase(self,text):
        return text.lower()

    def remove_punctuation(self,text):
        return ''.join(d for d in text if d not in string.punctuation)

    def remove_stopwords(self,text):
        stopwords = []
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            for word in f:
                word = word.split('\n')
                stopwords.append(word[0])
        clean_text = ' '.join(s for s in text.split() if s not in stopwords)
        return clean_text

    def remove_numbers(self,text):
        text = re.sub(r'\d', '', text)
        return text

    def remove_less_than_2(self,text):
        text = ' '.join([w for w in text.split() if len(w) > 2])
        return text

    def remove_extra_space(self,text):
        ornek_text_strip = re.sub(' +', ' ', text)
        return ornek_text_strip.strip()
