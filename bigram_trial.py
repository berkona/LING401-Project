
import nltk, random

from nltk.corpus import udhr

import pandas as pd

# what languages to look at
LANGUAGES = [ 
	"English-Latin1", 
	"French_Francais-Latin1",
	"Spanish-Latin1",
	"Italian-Latin1",
	"German_Deutsch-Latin1"
]


def wordProb(word, grammar):
	"""
	This is our actual metric implemented on a word basis
	"""
	assert word.isalpha()

	word = word.lower() + '$'
	p = 1.0
	last_char = '$'
	for i in range(len(word)):
		p += grammar[last_char].freq(word[i])
		last_char = word[i]
	return p / len(word)


def predictLanguage(word, grammars):
	"""
	A driver method for 1 word and all bigram grammars
	"""
	return [ ( wordProb(word, grammar), lang ) for lang, grammar in grammars ]


def predictSentLanguage(sent, grammars):
	"""
	A driver method for a sentence and all bigram grammars 
	"""
	return [ ( sum([ wordProb(word, grammar) for word in sent ]) / len(sent), lang ) for lang, grammar in grammars ] 


def makeBigramGrammars(all_bigrams):
	"""
	Construct a bigram grammar from a set of (lang, bigrams)
	"""
	return [
		( lang, nltk.ConditionalFreqDist(bigrams) )
		for lang, bigrams in all_bigrams
	]


def makeBigrams(words):
	"""
	This takes a text and creates all case-insentive bigrams for all alphabetic words in the given text
	"""
	return nltk.bigrams( "$" + "$".join(w.lower() for w in words if w.isalpha()) + "$" )


def filterWords(words):
	"""
	Just a helper function
	"""
	return ( w for w in words if w.isalpha() )


def runLeaveOutWordTrial(language):
	"""
	Choses a single word to exclude from the tokens of the UDHR and then tests against that
	"""
	all_words = list(filterWords(udhr.words(language)))
	test_set = random.choice(all_words)
	train_set = [ w for w in all_words if w not in test_set ]

	bigrams = [ (language, makeBigrams(train_set)) ]
	for lang in LANGUAGES:
		if lang == language:
			continue
		bigrams.append( ( lang, makeBigrams(udhr.words(lang)) ) )

	grammars = makeBigramGrammars(bigrams)
	
	return [ test_set ], predictLanguage(test_set, grammars)


def runLeaveOutWordTrialUnbiased(language):
	"""
	Choses a single word to exclude from the types of the UDHR and then tests against that
	"""
	all_words = list(set(filterWords(udhr.words(language))))
	test_set = random.choice(all_words)
	train_set = [ w for w in all_words if w not in test_set ]

	bigrams = [ (language, makeBigrams(train_set)) ]
	for lang in LANGUAGES:
		if lang == language:
			continue
		bigrams.append( ( lang, makeBigrams(udhr.words(lang)) ) )

	grammars = makeBigramGrammars(bigrams)
	
	return [ test_set ], predictLanguage(test_set, grammars)


def runLeaveOutSentTrial(language):
	"""
	Choses a sentence to exclude from the UDHR and then tests against that
	"""
	sents = udhr.sents(language)
	sent_idx = random.randint(0, len(sents)-1)
	test_sent = list(filterWords(sents[sent_idx]))
	train_set = []
	for i in range(len(sents)):
		if i == sent_idx:
			continue
		train_set += filterWords(sents[i])

	bigrams = [ (language, makeBigrams(train_set)) ]
	for lang in LANGUAGES:
		if lang == language:
			continue
		bigrams.append( ( lang, makeBigrams(udhr.words(lang)) ) )

	grammars = makeBigramGrammars(bigrams)
	
	return test_sent, predictSentLanguage(test_sent, grammars)


def runTrial(trialFn, n=100):
	"""
	Run a 'trial' some number of times (defaults to 100) and then return a DataFrame with full results
	"""
	results = pd.DataFrame(columns=[ 'Test Set', 'Expected', 'Actual', ] + LANGUAGES)
	for i in range(n):
		for lang in LANGUAGES:
			test_set, result = trialFn(lang)
			prediction = max(result)[1]
			results.loc[str(i) + "-" + lang] = [ " ".join(test_set), lang, prediction,  ] + [ p for p, l in result ] 
	return results


def accuracy(results):
	"""
	Helper to get accuracy from the results DataFrame
	"""
	n = results.groupby('Expected')['Expected'].count()[0]
	correct = results['Expected'] == results['Actual']
	return results[correct].groupby('Expected')['Expected'].count() / n


all_bigrams = [ ( lang, makeBigrams(udhr.words(lang)) ) for lang in LANGUAGES ]
bigram_grammars = makeBigramGrammars(all_bigrams)


def lengthTrial():
	"""
	A seperate trial that sorts all the words by length and then examines the accuracy on a per-length basis
	"""
	results = pd.DataFrame(columns=['Language', 'Length', 'Accuracy'])
	for lang in LANGUAGES:
		words_by_length = {}
		for word in set(filterWords(udhr.words(lang))):
			words_by_length[len(word)] = [ word ] + words_by_length.get(len(word), [])

		for l, words in words_by_length.items():
			correct = 0
			for w in words:
				result = predictLanguage(w, bigram_grammars)
				prediction = max(result)[1]
				correct += 1 if prediction == lang else 0
			accuracy = correct / len(words)
			results.loc[str(l) + "-" + lang] = [ lang, l, accuracy ]
	return results


def main():
	"""
	Run all trials and save to csv files
	"""
	length_results = lengthTrial()
	length_results.to_csv('bigram_results/length_trial.csv', index_label="index")

	biased_results = runTrial(runLeaveOutWordTrial)
	biased_accuracy = accuracy(biased_results)
	print(biased_accuracy)
	biased_results.to_csv('bigram_results/biased_trial.csv', index_label="index")

	unbiased_results = runTrial(runLeaveOutWordTrialUnbiased)
	unbiased_accuracy = accuracy(unbiased_results)
	print(unbiased_accuracy)
	unbiased_results.to_csv('bigram_results/unbiased_trial.csv', index_label="index")

	sent_results = runTrial(runLeaveOutSentTrial)
	sent_accuracy = accuracy(sent_results)
	print(sent_accuracy)
	sent_results.to_csv('bigram_results/sent_trial.csv', index_label="index")


if __name__ == '__main__':
	main()