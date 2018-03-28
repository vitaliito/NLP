from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
	"""
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure

	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.

			LM['house']['maison'] = 0.5
	"""
	AM = {}

	# Read training data

	sentences_e, sentences_f = read_hansard(train_dir, num_sentences)

    # Initialize AM uniformly

	AM = initialize(sentences_e, sentences_f)
    # Iterate between E and M steps

	iteration = 0

	while iteration < max_iter:
		AM = em_step(AM, sentences_e, sentences_f)
		iteration += 1

	AM["SENTEND"] = {}
	AM["SENTSTART"] = {}
	AM["SENTEND"]["SENTEND"] = 1
	AM["SENTSTART"]["SENTSTART"] = 1
	with open(fn_AM+'.pickle', 'wb') as handle:
		pickle.dump(AM, handle, protocol = pickle.HIGHEST_PROTOCOL)
	
	return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
	"""
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider


	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.
	"""
	filenames_e = list(glob.glob(os.path.join(train_dir, "*.e")))
	
	out_e = []
	out_f = []
	for file_e in filenames_e:
		
		file_f = file_e[:-1] + "f"
		
		with open(file_e) as f_e, open(file_f) as f_f:
			sentences_e = f_e.readlines()
			sentences_f = f_f.readlines()

			i = 0
			while i < num_sentences and i < len(sentences_e):
				e_s = preprocess(sentences_e[i].strip(), "e")
				f_s = preprocess(sentences_f[i].strip(), "f")
				e_s = " ".join((e_s.split())[1:-1])
				f_s = " ".join((f_s.split())[1:-1])
				out_e.append(e_s)
				out_f.append(f_s)
				i += 1
	
	return out_e, out_f

def initialize(eng, fre):
	"""
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	AM = {}
	
	for s in range(len(eng)):
		eng_s = eng[s].split()
		fre_s = fre[s].split()

		for e_w in eng_s:
			if not e_w in AM:
				AM[e_w] = {}
			for f_w in fre_s:
				AM[e_w][f_w] = 1/len(fre_s)


	return AM
    
def em_step(t, eng, fre):
	"""
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	tcount = {}
	total = {}
	for s in range(len(eng)):
		eng_s = eng[s].split()
		fre_s = fre[s].split()

		for e_w in eng_s:
			total[e_w] = 0
			if not e_w in tcount:
				tcount[e_w] = {}
			for f_w in fre_s:
				tcount[e_w][f_w] = 0
	for E, F in zip(eng, fre):
		E_u = set(E.split())
		F_u = set(F.split())
		for f in F_u:
			denom_c = 0
			for e in E_u:
				denom_c += t[e][f] * F.count(f)
			for e in E_u:
				tcount[e][f] += t[e][f] * F.count(f) * E.count(e) / denom_c
				total[e] += t[e][f] * F.count(f) * E.count(e) / denom_c
	for e in total:
		for f in tcount[e]:
			t[e][f] = tcount[e][f] / total[e]
			
	return t
