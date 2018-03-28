import math

def BLEU_score(candidate, references, n):
	"""
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

	candidate = candidate.split()

	p_1 = 0
	p_2 = 0
	p_3 = 0

	if n >= 1:
		for word in candidate:
			count = 0
			i = 0
			while count == 0 and i < len(references):
				count = int(word in references[i].split())
				i += 1
			p_1 += count
		p_1 = p_1 / len(candidate)

	if n >= 2:
		group_2_w = group_words(candidate, 2)
		for group in group_2_w:
			count = 0
			i = 0
			while count == 0 and i < len(references):
				count = int(group in group_words(references[i].split(), 2))
				i += 1
			p_2 += count
		p_2 = p_2 / len(group_2_w)

	if n >= 3:
		group_3_w = group_words(candidate, 3)
		for group in group_3_w:
			count = 0
			i = 0
			while count == 0 and i < len(references):
				count = int(group in group_words(references[i].split(), 3))
				i += 1
			p_3 += count
		p_3 = p_3 / len(group_3_w)

	brevity = closest_length(len(candidate), references) / len(candidate)
	if brevity < 1:
		BP_c = 1
	else:
		BP_c = math.exp(1 - brevity)
	
	if n == 1:
		bleu_score = BP_c * p_1
	if n == 2:
		bleu_score = BP_c * ((p_1 * p_2)**(1/n))
	if n == 3:
		bleu_score = BP_c * ((p_1 * p_2 * p_3)**(1/n))

	return bleu_score	

def group_words(sentence, n):
	"""
	Generates from a given list of words, a new list of grouped words 
	of size n.
	"""
	out = []
	for i in range(len(sentence)):
		if i + n - 1 < len(sentence):
			out.append(" ".join(sentence[i:i + n]))
	return out

def closest_length(cand_length, references):
	"""
	Returns the closest length to the candidate sentence length from
	the given reference lengths.
	"""
	if abs(cand_length - len(references[0].split())) <= abs(cand_length - len(references[1].split())):
		return len(references[0].split())
	else:
		return len(references[1].split())