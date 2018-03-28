from preprocess import *
import pickle
import os
import glob
import collections
import preprocess as prep

def lm_train(data_dir, language, fn_LM):
	"""
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
	data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
	"""
	filenames = list(glob.glob(os.path.join(data_dir, "*.{}".format(language))))
	language_model = {}
	language_model['uni'] = {}
	language_model['bi'] = {}

	for fname in filenames:
		with open(fname) as f:
			for line in f:
				processed = line.strip()
				processed = prep.preprocess(processed, language)
				processed = processed.split(" ")
				for i in range(len(processed)):
					if processed[i] in language_model['bi'] and i + 1 < len(processed):
						if processed[i + 1] in language_model['bi'][processed[i]]:
							language_model['bi'][processed[i]][processed[i + 1]] += 1
						else:
							language_model['bi'][processed[i]][processed[i + 1]] = 1
					if processed[i] in language_model['uni']:
						language_model['uni'][processed[i]] += 1
					else:
						language_model['bi'][processed[i]] = {}
						if i + 1 < len(processed):
							language_model['bi'][processed[i]][processed[i + 1]] = 1
						if processed[i] not in language_model['uni']:
							language_model['uni'][processed[i]] = 1

    #Save Model
	with open(fn_LM+'.pickle', 'wb') as handle:
		pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
	return language_model

if __name__ == "__main__":
	data_dir = "/Users/vitalytopekha/Desktop/CSC401/A2/data/Hansard/Training/"
	fn_LM = "fn_LM"

	#print(lm_train(data_dir, "e", fn_LM + "_e"))
	#print("Finished English Model")

	print(lm_train(data_dir, "f", fn_LM + "_f"))
	print("Finished French Model")