
import nltk

class NGramClassifier(object):

	def __init__(self, n, training_set):
		assert n > 1

		self.n = n
		self.padding = "$" * (self.n - 1)
		self._grammars = [
			( lang, self._makeGrammar(words) ) for lang, words in training_set
		]

	def classifyWord(self, word):
		return [ ( self._wordProb(word, grammar), lang ) for lang, grammar in self._grammars ]

	def classifySent(self, sent):
		sent = [ word for word in sent if word.isalpha() ]
		return [ ( sum([ self._wordProb(word, grammar) for word in sent ]) / len(sent), lang ) for lang, grammar in self._grammars ] 

	def _wordProb(self, word, grammar):
		assert word.isalpha() and len(word) > 0

		word = self.padding + word.lower() + self.padding
		k = 0.0
		p = 0.0
		for ngram in nltk.ngrams(word, self.n):
			p += grammar[ngram[:self.n-1]].freq(ngram[self.n-1])
			k += 1.0
		return p / k

	def _makeGrammar(self, words):
		words = ( w.lower() for w in words if w.isalpha() )
		all_ngrams = nltk.ngrams( ( self.padding + self.padding.join(words) + self.padding ), self.n )
		conditions = ( ( ngram[:self.n-1], ngram[self.n-1] ) for ngram in all_ngrams )
		return nltk.ConditionalFreqDist(conditions)

