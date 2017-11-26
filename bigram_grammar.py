
import nltk, random

from nltk.corpus import udhr

# [ x for x in udhr.fileids() if x.endswith("Latin1") ]

def wordProb(word, grammar):
	word = word.lower() + '$'
	p = 1.0
	last_char = '$'
	for i in range(len(word)):
		p += grammar[last_char].freq(word[i])
		last_char = word[i]
	return p / len(word)


def predictLanguage(word, grammars):
	return [ ( wordProb(word, grammar), lang ) for lang, grammar in grammars ]


def predictSentLanguage(sent, grammars):
	return [ ( sum([ wordProb(word, grammar) for word in sent ]), lang ) for lang, grammar in grammars ] 


def makeBigramGrammars(all_bigrams):
	return [
		( lang, nltk.ConditionalFreqDist(bigrams) )
		for lang, bigrams in all_bigrams
	]


def makeBigrams(words):
	return nltk.bigrams( "$" + "$".join(w.lower() for w in words if w.isalpha()) + "$" )


LANGUAGES = [ 
	"English-Latin1", 
	"French_Francais-Latin1",
	"Spanish-Latin1",
	"Italian-Latin1",
	"German_Deutsch-Latin1"
]

all_bigrams = [ ( lang, makeBigrams(udhr.words(lang)) ) for lang in LANGUAGES ]

bigram_grammars = makeBigramGrammars(all_bigrams)

def runLeaveOutWordTrial(language, n_test=1):
	all_words = list(udhr.words(language))
	test_set = random.sample(all_words, n_test)
	train_set = [ w for w in all_words if w not in test_set ]

	bigrams = [ (language, makeBigrams(train_set)) ]
	for lang in LANGUAGES:
		if lang == language:
			continue
		bigrams.append( ( lang, makeBigrams(udhr.words(lang)) ) )

	grammars = makeBigramGrammars(bigrams)
	correct = 0
	for word in test_set:
		result = predictLanguage(word, grammars)
		if (max(result)[1] == language):
			correct += 1
	return correct


accuracy = {}
for i in range(100):
	for l in LANGUAGES:
		accuracy[l] = runLeaveOutWordTrial(l, 1) + accuracy.get(l, 0)

print(accuracy)
