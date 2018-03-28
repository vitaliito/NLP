import numpy as np
import sys
import argparse
import os
import json
import regex as re
import math
import csv
import time

first_person_set = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
second_person_set = {"you", "your", "yours", "u", "ur", "urs"}
third_person_set = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}
slang_set = {"smh", "fwb", "lmfao", "lmao", "lms", "tbh", "rofl", "wtf", "bff", "wyd", "lylc", "brb",
                "atm", "imao", "sml", "btw", "imho", "fyi", "ppl", "sob", "ttyl", "imo", "ltr", "thx",
                "kk", "omg", "ttys", "afn", "bbs", "cya", "ez", "f2f", "gtr", "ic", "jk", "k", "ly",
                "ya", "nm", "np", "plz", "ru", "so", "tc", "tmi", "ym", "ur", "u", "sol", "lol", "fml"}
future_tense_set = {"’ll", "will", "gonna"}
common_noun_set = {"NN", "NNS"}
proper_noun_set = {"NNP", "NNPS"}
wh_word_set = {"WDT", "WP", "WP$", "WRB"}
adverb_set = {"RB", "RBR", "RBS"}
punct_tag_set = {"#", "$", ".", ",", ":", "(", ")", "\"", "‘", "“", "’", "”", "\'\'", "``"}
cat_to_int = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

brist_warr = np.zeros(12)

def bristol_warriner():
    ''' This function calculates features 18-29
    '''
    AoA_sum = 0
    IMG_sum = 0
    FAM_sum = 0

    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        total = 0
        for row in reader:
            if row[3] == '':
                break
            if row[3] != 'NA':
                AoA_sum += float(row[3])
            IMG_sum += float(row[4])
            FAM_sum += float(row[5])
            total += 1
    
    AoA_avg = AoA_sum / total
    IMG_avg = IMG_sum / total
    FAM_avg = IMG_sum / total

    brist_warr[0] = AoA_avg
    brist_warr[1] = IMG_avg
    brist_warr[2] = FAM_avg

    #calculate (x-u)^2 where u is sample mean
    AoA_sqr_diff = 0
    IMG_sqr_diff = 0
    FAM_sqr_diff = 0

    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        total = 0
        for row in reader:
            if row[3] == '':
                break
            if row[3] != 'NA':
                AoA_sqr_diff += (float(row[3]) - AoA_avg) ** 2
            IMG_sqr_diff += (float(row[4]) - IMG_avg) ** 2
            FAM_sqr_diff += (float(row[5]) - FAM_avg) ** 2
            total += 1

    # calculate standard deviation
    AoA_sd = math.sqrt(AoA_sqr_diff / (total - 1))
    IMG_sd = math.sqrt(IMG_sqr_diff / (total - 1))
    FAM_sd = math.sqrt(FAM_sqr_diff / (total - 1))

    brist_warr[3] = AoA_sd
    brist_warr[4] = IMG_sd
    brist_warr[5] = FAM_sd

    #STEPS: 24-29
    Vmean_sum = 0
    Amean_sum = 0
    Dmean_sum = 0

    with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        total = 0
        for row in reader:
            if row[2] == '':
                break
            Vmean_sum += float(row[2])
            Amean_sum += float(row[5])
            Dmean_sum += float(row[8])
            total += 1
    
    Vmean_avg = Vmean_sum / total
    Amean_avg = Amean_sum / total
    Dmean_avg = Dmean_sum / total

    brist_warr[6] = Vmean_avg
    brist_warr[7] = Amean_avg
    brist_warr[8] = Dmean_avg

    #calculate (x-u)^2 where u is sample mean
    Vmean_sqr_diff = 0
    Amean_sqr_diff = 0
    Dmean_sqr_diff = 0

    with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        total = 0
        for row in reader:
            if row[2] == '':
                break
            Vmean_sqr_diff += (float(row[2]) - Vmean_avg) ** 2
            Amean_sqr_diff += (float(row[5]) - Amean_avg) ** 2
            Dmean_sqr_diff += (float(row[8]) - Dmean_avg) ** 2
            total += 1

    # calculate standard deviation
    Vmean_sd = math.sqrt(Vmean_sqr_diff / (total - 1))
    Amean_sd = math.sqrt(Amean_sqr_diff / (total - 1))
    Dmean_sd = math.sqrt(Dmean_sqr_diff / (total - 1))

    brist_warr[9] = Vmean_sd
    brist_warr[10] = Amean_sd
    brist_warr[11] = Dmean_sd



def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros(173+1)
    comment_split = comment.split(" ")
    count_ends = sum([1 for i in comment_split if "\n" in i])
    punct_tag_count = 0
    for i in range(len(comment_split)):
        index = comment_split[i].rfind("/")
        tag = comment_split[i][index+1:]
        word = comment_split[i][0:index]
    #STEP 1. Number of first-person pronouns
        feats[0] += int(word in first_person_set)
    #STEP 2. Number of second-person pronouns
        feats[1] += int(word in second_person_set)
    #STEP 3. Number of third-person pronouns
        feats[2] += int(word in third_person_set)
    #STEP 4. Number of coordinating conjunctions
        feats[3] += tag.count("CC")
    #STEP 5. Number of past-tense verbs
        feats[4] += tag.count("VBD")
    #STEP 6. Number of future-tense verbs
        if word == "to":
            # if word is "to", check next tag to be VB and previous word to be "going"
            if i > 0 and i < (len(comment_split) - 1):
                pre = comment_split[i-1]
                nxt = comment_split[i+1]
                pre_word = pre[0:pre.rfind("/")]
                nxt_tag = nxt[nxt.rfind("/")+1:]
                if pre_word == "going" and nxt_tag == "VB":
                    feats[5] += 1
        feats[5] += int(word in future_tense_set)
    #STEP 7. Number of commas
        feats[6] += tag.count(",")
    #STEP 8. Number of multi-character punctuation tokens
        pattern_any_punctuation = re.compile(r'(?:[!#\$%&\(\)\*\+,\./:;<=>\?@\[\\\]\^_`\{\|\}~]{2,})')
        match = pattern_any_punctuation.search(word)
        feats[7] += int(match)
    #STEP 9. Number of common nouns
        feats[8] += int(tag in common_noun_set)
    #STEP 10. Number of proper nouns
        feats[9] += int(tag in proper_noun_set)
    #STEP 11. Number of adverbs
        feats[10] += int(tag in adverb_set)
    #STEP 12. Number of wh- words
        feats[11] += int(tag in wh_word_set)
    #STEP 13. Number of slang acronyms
        feats[12] += int(word in slang_set)
    #STEP 14. Number of words in uppercase (≥ 3 letters long)
        feats[13] += int(word.isupper() and len(word) >= 3)
    #STEP 15. Average length of sentences, in tokens
        if count_ends == 0:
            count_ends = 1
        feats[14] = len(comment_split) / count_ends
    #STEP 16. Average length of tokens, excluding punctuation-only tokens, in characters
        if tag in punct_tag_set:
            punct_tag_count += 1
        feats[15] = len(comment_split) - punct_tag_count
    #STEP 17. Number of sentences
        feats[16] = count_ends
    #STEPS 18-29:
        if not np.any(brist_warr):
            bristol_warriner()
        feats[17:29] = brist_warr
    #STEP 30-173. LIWC/Receptiviti features

    return feats

def main( args ):
    
    path_to_feats = "/u/cs401/A1/feats"
    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))
    alt_data = np.load(path_to_feats + '/Alt_feats.dat.npy')
    center_data = np.load(path_to_feats + '/Center_feats.dat.npy')
    left_data = np.load(path_to_feats + '/Left_feats.dat.npy')
    right_data = np.load(path_to_feats + '/Right_feats.dat.npy')

    left_id_i = {}
    center_id_i = {}
    right_id_i = {}
    alt_id_i = {}

    with open(path_to_feats + "/Left_IDs.txt") as f:
        index = 0
        for line in f:
            left_id_i[line.strip()] = index
            index += 1
    with open(path_to_feats + "/Center_IDs.txt") as f:
        index = 0
        for line in f:
            center_id_i[line.strip()] = index
            index += 1
    with open(path_to_feats + "/Right_IDs.txt") as f:
        index = 0
        for line in f:
            right_id_i[line.strip()] = index
            index += 1
    with open(path_to_feats + "/Alt_IDs.txt") as f:
        index = 0
        for line in f:
            alt_id_i[line.strip()] = index
            index += 1



    for i in range(len(data)):
        comment = json.loads(data[i])
        feats[i] = extract1(comment["body"])
        feats[i][-1] = cat_to_int[comment["cat"]]
        if feats[i][-1] == 0:
            feats[i][29:173] = left_data[left_id_i[comment["id"]]]
        elif feats[i][-1] == 1:
            feats[i][29:173] = center_data[center_id_i[comment["id"]]]
        elif feats[i][-1] == 2:
            feats[i][29:173] = right_data[right_id_i[comment["id"]]]
        elif feats[i][-1] == 3:
            feats[i][29:173] = alt_data[alt_id_i[comment["id"]]]
       
    np.savez_compressed( args.output, feats)
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

