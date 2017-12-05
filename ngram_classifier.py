
import nltk

class NGramClassifier(object):
	"""
	This is our final classifier.
	classifyWord and classifySent both return a sequence of tuples of (lang, certainity) for all trained languages
	"""

	def __init__(self, n, training_set):
		"""
		Params
		- n, integer -- order of the N-gram model
		- training_set, sequence -- data to train on.  Each element should be a 2-tuple of:
			- position 0, string -- what language these words belong to (should be unique)
			- position 1, sequence of strings -- words to train for that language, non-alpha words will be ignored.
		"""
		assert n > 1

		self.n = n
		self.padding = "$" * (self.n - 1)
		self._grammars = [
			( lang, self._makeGrammar(words) ) for lang, words in training_set
		]

	def classifyWord(self, word):
		"""
		Classify a single word using the training data
		"""
		return [ ( self._wordProb(word, grammar), lang ) for lang, grammar in self._grammars ]

	def classifySent(self, sent):
		"""
		Classify a sentence (or any sequence of words) using the training data
		"""
		sent = [ word for word in sent if word.isalpha() ]
		return [ ( sum([ self._wordProb(word, grammar) for word in sent ]) / len(sent), lang ) for lang, grammar in self._grammars ] 

	def _wordProb(self, word, grammar):
		"""
		Internal helper function to get our metric for a given word and grammar
		"""
		assert word.isalpha() and len(word) > 0

		word = self.padding + word.lower() + self.padding
		k = 0.0
		p = 0.0
		for ngram in nltk.ngrams(word, self.n):
			p += grammar[ngram[:self.n-1]].freq(ngram[self.n-1])
			k += 1.0
		return p / k

	def _makeGrammar(self, words):
		"""
		Internal helper function to construct a n-gram grammar from a set of words
		"""
		words = ( w.lower() for w in words if w.isalpha() )
		all_ngrams = nltk.ngrams( ( self.padding + self.padding.join(words) + self.padding ), self.n )
		conditions = ( ( ngram[:self.n-1], ngram[self.n-1] ) for ngram in all_ngrams )
		return nltk.ConditionalFreqDist(conditions)

