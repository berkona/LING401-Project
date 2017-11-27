
import nltk, random

from nltk.corpus import udhr

import pandas as pd

from ngram_classifier import NGramClassifier

LANGUAGES = [ 
	"English-Latin1", 
	"French_Francais-Latin1",
	"Spanish-Latin1",
	"Italian-Latin1",
	"German_Deutsch-Latin1"
]

N = 5

def filterWords(words):
	return ( w for w in words if w.isalpha() )


def runLeaveOutWordTrial(language):
	all_words = list(filterWords(udhr.words(language)))
	test_set = random.choice(all_words)
	train_set = [ w for w in all_words if w not in test_set ]

	ngrams = [ ( language, train_set ) ]
	for lang in LANGUAGES:
		if lang == language:
			continue
		ngrams.append( ( lang, udhr.words(lang) ) )

	classifier = NGramClassifier(N, ngrams)
	
	return [ test_set ], classifier.classifyWord(test_set)


def runLeaveOutWordTrialUnbiased(language):
	all_words = list(set(filterWords(udhr.words(language))))
	test_set = random.choice(all_words)
	train_set = [ w for w in all_words if w not in test_set ]

	ngrams = [ ( language, train_set ) ]
	for lang in LANGUAGES:
		if lang == language:
			continue
		ngrams.append( ( lang, udhr.words(lang) ) )

	classifier = NGramClassifier(N, ngrams)
	
	return [ test_set ], classifier.classifyWord(test_set)


def runLeaveOutSentTrial(language):
	sents = udhr.sents(language)
	sent_idx = random.randint(0, len(sents)-1)
	test_sent = sents[sent_idx]
	train_set = []
	for i in range(len(sents)):
		if i == sent_idx:
			continue
		train_set += sents[i]

	ngrams = [ ( language, train_set ) ]
	for lang in LANGUAGES:
		if lang == language:
			continue
		ngrams.append( ( lang, udhr.words(lang) ) )

	classifier = NGramClassifier(N, ngrams)
	
	return test_sent, classifier.classifySent(test_sent)


def runTrial(trialFn, n=100):
	results = pd.DataFrame(columns=[ 'Test Set', 'Expected', 'Actual', ] + LANGUAGES)
	for i in range(n):
		for lang in LANGUAGES:
			test_set, result = trialFn(lang)
			prediction = max(result)[1]
			results.loc[str(i) + "-" + lang] = [ " ".join(test_set), lang, prediction,  ] + [ p for p, l in result ] 
	return results


def accuracy(results):
	n = results.groupby('Expected')['Expected'].count()[0]
	correct = results['Expected'] == results['Actual']
	return results[correct].groupby('Expected')['Expected'].count() / n


all_bigrams = [ ( lang, udhr.words(lang) ) for lang in LANGUAGES ]
ngram_classifier = NGramClassifier(N, all_bigrams)
# bigram_grammars = makeBigramGrammars(all_bigrams)


def lengthTrial():
	results = pd.DataFrame(columns=['Language', 'Length', 'Accuracy'])
	for lang in LANGUAGES:
		words_by_length = {}
		for word in set(filterWords(udhr.words(lang))):
			words_by_length[len(word)] = [ word ] + words_by_length.get(len(word), [])

		for l, words in words_by_length.items():
			correct = 0
			for w in words:
				result = ngram_classifier.classifyWord(w)
				prediction = max(result)[1]
				correct += 1 if prediction == lang else 0
			accuracy = correct / len(words)
			results.loc[str(l) + "-" + lang] = [ lang, l, accuracy ]
	return results


def main():
	length_results = lengthTrial()
	length_results.to_csv('ngram_results/length_trial.csv', index_label="index")

	biased_results = runTrial(runLeaveOutWordTrial)
	biased_accuracy = accuracy(biased_results)
	print(biased_accuracy)
	biased_results.to_csv('ngram_results/biased_trial.csv', index_label="index")

	unbiased_results = runTrial(runLeaveOutWordTrialUnbiased)
	unbiased_accuracy = accuracy(unbiased_results)
	print(unbiased_accuracy)
	unbiased_results.to_csv('ngram_results/unbiased_trial.csv', index_label="index")

	sent_results = runTrial(runLeaveOutSentTrial)
	sent_accuracy = accuracy(sent_results)
	print(sent_accuracy)
	sent_results.to_csv('ngram_results/sent_trial.csv', index_label="index")


if __name__ == '__main__':
	main()