from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
	"""
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	log_prob = 0

	words = sentence.split(" ")
	for i in range(len(words)):
		if i + 1 < len(words):
			if words[i] in LM['bi'] and words[i + 1] in LM['bi'][words[i]]:
				count_w1_w2 = LM['bi'][words[i]][words[i + 1]]
				count_w1 = LM['uni'][words[i]]
				numerator = count_w1_w2 + delta
				denominator = count_w1 + delta * vocabSize
				if denominator == 0 and numerator == 0:
					log_prob += float("-inf") 
				else:
					log_prob += log(numerator/denominator, 2)



            
	return log_prob