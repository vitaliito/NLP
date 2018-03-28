from log_prob import *
from preprocess import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
    Computes the preplexity of language model given a test corpus

    INPUT:

    LM : 		(dictionary) the language model trained by lm_train
    test_dir : 	(string) The top-level directory name containing data
    			e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    language : `(string) either 'e' (English) or 'f' (French)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta : 	(float) smoothing parameter where 0<delta<=1
    """
	
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])

    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue

        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)

            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
    	pp = 2**(-pp/N)
    return pp
if __name__ == "__main__":
    #test
    test_dir = "/Users/vitalytopekha/Desktop/CSC401/A2/data/Hansard/Testing/"
    train_dir = "/Users/vitalytopekha/Desktop/CSC401/A2/data/Hansard/Training/"

    LM_e = lm_train(train_dir, "e", "fn_LM_e")
    LM_f = lm_train(train_dir, "f", "fn_LM_f")

    # compute perplexity for English Language Model
    language = 'e'
    print('English Language Model (MLE) :', preplexity(LM_e, test_dir, language, smoothing = False, delta = 0))
    print('English Language Model (delta = 0.0001) :', preplexity(LM_e, test_dir, language, smoothing = True, delta = 0.0001, vocabSize = len(LM_e['uni'])))
    print('English Language Model (delta = 0.001) :', preplexity(LM_e, test_dir, language, smoothing = True, delta = 0.001, vocabSize = len(LM_e['uni'])))
    print('English Language Model (delta = 0.01) :', preplexity(LM_e, test_dir, language, smoothing = True, delta = 0.01, vocabSize = len(LM_e['uni'])))
    print('English Language Model (delta = 0.1) :', preplexity(LM_e, test_dir, language, smoothing = True, delta = 0.1, vocabSize = len(LM_e['uni'])))
    
    # compute perplexity for French Language Model
    language = 'f'
    print('French Language Model (MLE) :', preplexity(LM_f, test_dir, language, smoothing = False, delta = 0))
    print('French Language Model (delta = 0.0001) :', preplexity(LM_f, test_dir, language, smoothing = True, delta = 0.0001, vocabSize = len(LM_f['uni'])))
    print('French Language Model (delta = 0.001) :', preplexity(LM_f, test_dir, language, smoothing = True, delta = 0.001, vocabSize = len(LM_f['uni'])))
    print('French Language Model (delta = 0.01) :', preplexity(LM_f, test_dir, language, smoothing = True, delta = 0.01, vocabSize = len(LM_f['uni'])))
    print('French Language Model (delta = 0.1) :', preplexity(LM_f, test_dir, language, smoothing = True, delta = 0.1, vocabSize = len(LM_f['uni'])))