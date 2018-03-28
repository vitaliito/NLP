import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # For both languages, separate sentence-final punctuation (sentences have 
    # already been determined for you), commas, colons and semicolons, parentheses, 
    # dashes between parentheses, mathematical operators (e.g., +, -, <, >,
    # =), and quotation marks. Certain contractions are required in French, often 
    # to eliminate vowel clusters. When the input language is ‘french’, separate 
    # the following contractions: (le, la), (j'ai -> j' ai), (qu'on -> qu' on),
    # (puisqu'on -> puisqu' on)

    in_sentence = in_sentence.lower()
    if language == 'e':
        out_sentence = re.findall(r"[\w]+|[.,!?;\"/<>(){}\[\]\&\*\^\-\+]", in_sentence)
    else:
        out_sentence = re.findall(r"(?:j'|t'|l'|qu'|puisqu'|lorsqu')|[\w]+|[.,!?;\"/<>(){}\[\]\&\*\^\-\+]", in_sentence)
    out_sentence = ["SENTSTART"] + out_sentence + ["SENTEND"]
    out_sentence = " ".join(out_sentence)
    return out_sentence
