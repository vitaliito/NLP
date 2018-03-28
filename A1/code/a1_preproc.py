import sys
import argparse
import os
import json
import html
import re
import string
import numpy as np
import spacy

indir = '/u/cs401/A1/data'
abbrev_english_set = set(line.strip() for line in open('/u/cs401/Wordlists/abbrev.english'))
stopwords_set =  set(line.strip() for line in open('/u/cs401/Wordlists/StopWords'))
nlp = spacy.load('en', disable=['parser', 'ner'])

def preproc1(comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    if 1 in steps:
        # Remove all newline characters.
        modComm = comment.replace('\n', '')
    if 2 in steps:
        #  Replace HTML character codes (i.e., &...;) with their ASCII equivalent
        # (see http://www.asciitable.com).
        modComm = html.unescape(modComm)
    if 3 in steps:
        # Remove all URLs (i.e., tokens beginning with http or www).
        modComm = re.sub(r'http\S+', '', modComm)
        modComm = re.sub(r'www\S+', '', modComm)
    if 4 in steps:
        # Split each punctuation (see string.punctuation) into
        # its own token using whitespace except:
            # Apostrophes.
            # Periods in abbreviations (e.g., e.g.) are not split from their tokens. E.g., e.g. stays e.g.
            # Multiple punctuation (e.g., !?!, ...) are not split internally. E.g., Hi!!! becomes Hi !!!
            # You can handle single hyphens (-) between words as you please. E.g., you can split non-committal
            #    into three tokens or leave it as one.
        modComm = re.findall(r"(?:/Ala\.|Ariz\.|Assn\.|Atty\.|Aug\.|Ave\.|Bldg\.|Blvd\.|Calif\.|Capt\.|Cf\.|Ch\.|Co\.|Col\.|U.S.?|U.K.?"
        r"|Colo\.|Conn\.|Corp\.|DR\.|Dec\.|Dept\.|Dist\.|Dr\.|Drs\.|Ed\.|Eq\.|FEB\.|Feb\.|Fig\.|Figs\.|Fla\.|Ga\.|Gen\.|Gov\.|HON\.|Ill\."
        r"|Inc\.|JR\.|Jan\.|Jr\.|Kan\.|Ky\.|La\.|Lt\.|Ltd\.|MR\.|MRS\.|Mar\.|Mass\.|Md\.|Messrs\.|Mich\.|Minn\.|Miss\.|Mmes\.|Mo\.|Mr\.|Mrs\."
        r"|Ms\.|Mx\.|Mt\.|NO\.|No\.|Nov\.|Oct\.|Okla\.|Op\.|Ore\.|Pa\.|Pp\.|Prof\.|Prop\.|Rd\.|Ref\.|Rep\.|Reps\.|Rev\.|Rte\.|Sen\.|Sept\."
        r"|Sr\.|St\.|Stat\.|Supt\.|Tech\.|Tex\.|Va\.|Vol\.|Wash\.|al\.|av\.|ave\.|ca\.|cc\.|chap\.|cm\.|cu\.|dia\.|dr\.|eqn\.|etc\.|fig\."
        r"|figs\.|ft\.|gm\.|hr\.|in\.|kc\.|lb\.|lbs\.|mg\.|ml\.|mm\.|mv\.|nw\.|oz\.|pl\.|pp\.|sec\.|sq\.|st\.|vs\.|yr\.|e.g.|i.e.|a.i.|a.m."
        r"|A.M.|p.m.|P.M.|et al.|i.a.|Ph.D|R.I.P.|r.i.p.|vs.|S.O.S.|s.o.s.)|[\?\!\.]+|[$\w$%']+|[.,!?;\"/<>(){}\[\]\&\*\^\-\+]", modComm)
        if 5 not in steps:
            modComm = " ".join(modComm)
    if 5 in steps:
        # Split clitics using whitespace.
        # here we find indexes of words that have apostrophe in it.
        indexes = [i for i, s in enumerate(modComm) if "'" in s]
        modComm = np.array(modComm)
        updated = []
        prev = 0
        last = 0
        # go over each word with an apostrophe in it and match it with regular
        # expression that splits clitic and word. Then we make a new list with the
        # clitics separated from the word
        for i in indexes:
            if i >= 0:
                updated.extend(modComm[prev:i])
                if i < len(modComm):
                    prev = i + 1
            ext = re.findall(r"(?:[Cc]ould|[Dd]oes|[Dd]o|[Dd]id|[Hh]ave|[Hh]as|[Ww]ould|[Ww]ere|[Ww]as|[Aa]re|[Ii]s|[Cc]a)|(?:n?\'+\w*)|[\w]+", modComm[i])
            updated.extend(ext)
            last = i
        if last != 0 and last < len(modComm) + 1:
            updated.extend(modComm[last+1:len(modComm)])
        if not indexes:
            updated = modComm
        modComm = []
        for i in updated:
            modComm.append(str(i))
        if 6 not in steps:
            modComm = " ".join(modComm)
    if 6 in steps:
        # Each token is tagged with its part-of-speech using spaCy.
        modComm = spacy.tokens.Doc(nlp.vocab, words=modComm)
        tagged = nlp.tagger(modComm)
        modComm = []
        noTag = []
        for token in tagged:
            modComm.append(token.text + "/" + token.tag_)
            noTag.append(token.text)
        if 7 not in steps:
            modComm = " ".join(modComm)
    if 7 in steps:
        # Remove stopwords. See /u/cs401/Wordlcists/StopWords.
        tagged_i = [x for x in range(len(tagged)) if tagged[x].text not in stopwords_set]
        tagged = [tagged[i] for i in tagged_i]
        modComm = [modComm[i] for i in tagged_i]
        if 8 not in steps:
            modComm = " ".join(modComm)
    if 8 in steps:
        # Apply lemmatization using spaCy (see below).
        modComm = []
        for token in tagged:
            if token.lemma_[0] != "-":
                modComm.append(token.lemma_ + "/" + token.tag_)
            else:
                modComm.append(token.text + "/" + token.tag_)
        if 9 not in steps:
            modComm = " ".join(modComm)
    if 9 in steps:
        for i in range(len(modComm)):
            index = modComm[i].rfind("/")
            # if the tag of the token is . we know that it's an end of a sentence
            if modComm[i][-1] == ".":
                modComm[i] = modComm[i][0:index] + "\n" + modComm[i][index:]
            # if the word has a dot in it, it means that it's an abbreviation
            #   case 1. abbreviation in the end of the sentence we treat as ending punctuation
            #   case 2. abbreviation that is followed by a word, with first character in uppercase
            #           means that it is an end of a sentence.
            elif ("." in modComm[i][0:index]): 
                if (i + 1 < len(modComm)):
                    index_next = modComm[i + 1].rfind("/")
                    word = modComm[i + 1][0:index_next]
                    if word[0].isupper():
                        modComm[i] = modComm[i][0:index] + "\n" + modComm[i][index:]
                    elif i + 1 == len(modComm) - 1:
                        modComm[i] = modComm[i][0:index] + "\n" + modComm[i][index:]
        if 10 not in steps:
            modComm = " ".join(modComm)
    if 10 in steps:
        for i in range(len(modComm)):
            index = modComm[i].rfind("/")
            if index != -1:
                modComm[i] = modComm[i][0:index].lower() + modComm[i][index:]
            else:
                modComm[i] = modComm[i]
        modComm = " ".join(modComm)
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:

            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)
            
            data = json.load(open(fullFile))

            j = args.ID[0] % len(data)
            i = 0
            while i < args.max:
                if j == len(data):
                    j = 0
                line = data[j]    
                dictionary = json.loads(data[j])
                field_id = dictionary["id"] # choose to retain fields from those lines that are relevant to you 
                field_cat = file # add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                field_body = preproc1(dictionary["body"], range(1,11)) # process the body field (j['body']) with preproc1(...) using default for `steps` argument
                output = {}
                output["id"] = field_id
                output["cat"] = field_cat
                output["body"] = field_body # replace the 'body' field with the processed text
                json_output = json.dumps(output)
                allOutput.append(json_output)
                i += 1
                j += 1
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type = int,help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()
    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
        
    main(args)
