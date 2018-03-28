import math
import random
import numpy as np
from math import log
from preprocess import *
from lm_train import *
from log_prob import *
from align_ibm1 import * 
from decode import *
from BLEU_score import *

def evalAlign(max_iter):
    ''' 
    Translate the 25 French sentences in /u/cs401/A2 SMT/data/Hansard/Testing/Task5.f
    with the decode function and evaluate them using corresponding reference sentences,
    specifically:
    
    1. /u/cs401/A2 SMT/data/Hansard/Testing/Task5.e, from the Hansards.
    2. /u/cs401/A2 SMT/data/Hansard/Testing/Task5.google.e, Google’s translations of the French phrases2.
    
    To evaluate each translation, use the BLEU score from lecture 6,
    
    Repeat this task with at least four alignment models (trained on 1K, 10K, 15K, and 30K
    sentences, respectively) and with three values of n in the BLEU score (i.e., n = 1, 2, 
    3). You should therefore have 25×4×3 BLEU scores in your evaluation.
    '''

    bleu = np.zeros(shape = (25, 4, 3))

    train_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
    LM = lm_train(train_dir, "e", "fn_LM_e")
    num_sentences = [1000, 10000, 15000, 30000]
    for n in range(len(num_sentences)):

        n_s = num_sentences[n]
        AM = align_ibm1(train_dir, n_s, max_iter, "fm_AM_e_{}".format(n_s))

        with open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f") as candidate_sentences, open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e") as ref_1, open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e") as ref_2:
            candidate_sentences = candidate_sentences.readlines()
            ref_1 = ref_1.readlines()
            ref_2 = ref_2.readlines()
            for i in range(len(candidate_sentences)):
                sentence = candidate_sentences[i].strip()
                sentence = preprocess(sentence, "f")
                ref_1_sentence = preprocess(ref_1[i].strip(), "e")
                ref_2_sentence = preprocess(ref_2[i].strip(), "e")
                english = decode(sentence, LM, AM)
                bleu[i][n][0] = BLEU_score(english, [ref_1_sentence, ref_2_sentence], 1)
                bleu[i][n][1] = BLEU_score(english, [ref_1_sentence, ref_2_sentence], 2)
                bleu[i][n][2] = BLEU_score(english, [ref_1_sentence, ref_2_sentence], 3)
    return bleu

if __name__ == "__main__":
    iterations = [5, 10, 15, 20, 25]
    for iteration in iterations:
        bleu = evalAlign(iteration)

        with open("BLEU_score_{}".format(iteration) + '.pickle', 'wb') as handle:
            pickle.dump(bleu, handle, protocol = pickle.HIGHEST_PROTOCOL)
        