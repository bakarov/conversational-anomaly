from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from gensim.models import Word2Vec, KeyedVectors
from numpy import average, array

RU = 'ru'
EN = 'en'

class Vectorizer:

    def __init__(self, lang=RU):
        self._lang = lang
        self._tokenizer = self.set_tokenizer()
        self._stop = self.set_stop()
        self._lemmer = self.set_lemmer()
        self._model = self.set_w2v_model()
        self._vocab = self.set_w2v_vocab()

    def set_tokenizer(self):
        if self._lang == EN:
            return RegexpTokenizer('[A-Za-z]\w+')
        elif self._lang == RU:
            return RegexpTokenizer('[A-ZА-Яa-zа-я]\w+')


    def set_lemmer(self):
        if self._lang == EN:
            return WordNetLemmatizer()
        elif self._lang == RU:
            return MorphAnalyzer()


    def set_stop(self):
        if self._lang == EN:
            return stopwords.words('english')
        elif self._lang == RU:
            return stopwords.words('russian')


    def set_w2v_model(self):
        if self._lang == EN:
            return KeyedVectors.load_word2vec_format('models/google_news.bin', binary=True)
        elif self._lang == RU:
            return Word2Vec.load('models/2ch_model')

    def set_w2v_vocab(self):
        if self._lang == EN:
            return self._model.vocab
        elif self._lang == RU:
            return self._model.wv.vocab


    def __morph__(self, word):
        if self._lang == EN:
            return self._lemmer.lemmatize(word.lower())
        elif self._lang == RU:
            return self._lemmer.parse(word.lower())[0].normal_form

    def morph_sentence(self,    sent):
        return [self.__morph__(word) for word in self._tokenizer.tokenize(sent)
            if word.lower() not in self._stop]

    def make_vectors(self, t):
        t = [str(document) for document in t]
        k = [[self.__morph__(word) for word
              in self._tokenizer.tokenize(document) if word.lower()
              not in self._stop]
              for document in t]
        vectors = [[self._model[word] for word in sent if word in self._vocab] for sent in k]
        return array([average(vector,axis=0) for vector in vectors if vector])
