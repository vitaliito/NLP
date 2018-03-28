import numpy as np
import sys
import argparse
import os
import json
import regex as re
import math
import csv
import time

first_p_set = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
second_p_set = {"you", "your", "yours", "u", "ur", "urs"}
third_p_set = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}
future_tense_set = {"’ll", "will", "gonna"}
slang_set = {"smh", "fwb", "lmfao", "lmao", "lms", "tbh", "rofl", "wtf", "bff", "wyd", "lylc", "brb",
                "atm", "imao", "sml", "btw", "imho", "fyi", "ppl", "sob", "ttyl", "imo", "ltr", "thx",
                "kk", "omg", "ttys", "afn", "bbs", "cya", "ez", "f2f", "gtr", "ic", "jk", "k", "ly",
                "ya", "nm", "np", "plz", "ru", "so", "tc", "tmi", "ym", "ur", "u", "sol", "lol", "fml"}
proper_noun_set = {"NNP", "NNPS"}
common_noun_set = {"NN", "NNS"}
advb_set = {"RB", "RBR", "RBS"}
wh_word_set = {"WDT", "WP", "WP$", "WRB"}
punc_tag_set = {"#", "$", ".", ",", ":", "(", ")", "\"", "‘", "“", "’", "”", "\'\'", "``"}
cat_to_i = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

dir_feats = "/u/cs401/A1/feats"
alt_id = {}
left_id = {}
right_id = {}
center_id = {}
brist_w = np.zeros(12)

def onStart():

    sums = np.zeros(3)
    total = 0
    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        for row in reader:
            if row[3] == '':
                break
            if row[3] != 'NA':
                sums[0] += float(row[3])
            sums[1] += float(row[4])
            sums[2] += float(row[5])
            total += 1

    brist_w[0] = sums[0] / total
    brist_w[1] = sums[1] / total
    brist_w[2] = sums[2] / total

    #calculate (x-u)^2 where u is the mean of the sample
    sqr_diffs = np.zeros(3)
    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        total = 0
        for row in reader:
            if row[3] == '':
                break
            if row[3] != 'NA':
                sqr_diffs[0] += (float(row[3]) - brist_w[0]) ** 2
            sqr_diffs[1] += (float(row[4]) - brist_w[1]) ** 2
            sqr_diffs[2] += (float(row[5]) - brist_w[2]) ** 2
            total += 1

    # calculate standard deviation
    brist_w[3] = math.sqrt(sqr_diffs[0] / (total - 1))
    brist_w[4] = math.sqrt(sqr_diffs[1] / (total - 1))
    brist_w[5] = math.sqrt(sqr_diffs[2] / (total - 1))
    
    total = 0
    sums = np.zeros(3)

    with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        for row in reader:
            if row[2] == '':
                break
            sums[0] += float(row[2])
            sums[1] += float(row[5])
            sums[2] += float(row[8])
            total += 1

    brist_w[6] = sums[0] / total
    brist_w[7] = sums[1] / total
    brist_w[8] = sums[2] / total

    #calculate (x-u)^2 where u is sample mean
    sqr_diffs = np.zeros(3)
    total = 0

    with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        next(reader, None)
        for row in reader:
            if row[2] == '':
                break
            sqr_diffs[0] += (float(row[2]) - brist_w[6]) ** 2
            sqr_diffs[1] += (float(row[5]) - brist_w[7]) ** 2
            sqr_diffs[2] += (float(row[8]) - brist_w[8]) ** 2
            total += 1

    # calculate standard deviation
    brist_w[9] = math.sqrt(sqr_diffs[0] / (total - 1))
    brist_w[10] = math.sqrt(sqr_diffs[1] / (total - 1))
    brist_w[11] = math.sqrt(sqr_diffs[2] / (total - 1))

    with open(dir_feats + "/Left_IDs.txt") as f:
        index = 0
        for line in f:
            left_id[line.strip()] = index
            index += 1
    with open(dir_feats + "/Center_IDs.txt") as f:
        index = 0
        for line in f:
            center_id[line.strip()] = index
            index += 1
    with open(dir_feats + "/Right_IDs.txt") as f:
        index = 0
        for line in f:
            right_id[line.strip()] = index
            index += 1
    with open(dir_feats + "/Alt_IDs.txt") as f:
        index = 0
        for line in f:
            alt_id[line.strip()] = index
            index += 1

    return True

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
    #STEPS 1-17:
    for i in range(len(comment_split)):
        index = comment_split[i].rfind("/")
        tag = comment_split[i][index+1:]
        word = comment_split[i][0:index]
        if word in first_p_set:
            feats[0] += 1
        if word in second_p_set:
            feats[1] += 1
        if word in third_p_set:
            feats[2] += 1
        feats[3] += tag.count("CC")
        feats[4] += tag.count("VBD")
        if word == "to":
            if i > 0 and i < (len(comment_split) - 1):
                pre = comment_split[i-1]
                nxt = comment_split[i+1]
                pre_word = pre[0:pre.rfind("/")]
                nxt_tag = nxt[nxt.rfind("/")+1:]
                if pre_word == "going" and nxt_tag == "VB":
                    feats[5] += 1
        if word in future_tense_set:
            feats[5] += 1
        feats[6] += tag.count(",")
        pattern_any_punctuation = re.compile(r'(?:[!#\$%&\(\)\*\+,\./:;<=>\?@\[\\\]\^_`\{\|\}~]{2,})')
        match = pattern_any_punctuation.search(word)
        if match:
            feats[7] += 1
        if tag in common_noun_set:
            feats[8] += 1
        if tag in proper_noun_set:
            feats[9] += 1
        if tag in advb_set:
            feats[10] += 1
        if tag in wh_word_set:
            feats[11] += 1
        if word in slang_set:
            feats[12] += 1
        if word.isupper() and len(word) >= 3:
            feats[13] += 1
        if count_ends == 0:
            count_ends = 1
        feats[14] = len(comment_split) / count_ends
        if tag in punc_tag_set:
            punct_tag_count += 1
        feats[15] = len(comment_split) - punct_tag_count
    #STEP 17. Number of sentences
        feats[16] = count_ends
    #STEPS 18-29:
        feats[17:29] = brist_w
    #STEP 30-173. LIWC/Receptiviti features

    return feats

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    dir_feats = "/u/cs401/A1/feats"
    alt_data = np.load(dir_feats + '/Alt_feats.dat.npy')
    center_data = np.load(dir_feats + '/Center_feats.dat.npy')
    left_data = np.load(dir_feats + '/Left_feats.dat.npy')
    right_data = np.load(dir_feats + '/Right_feats.dat.npy')
    if onStart():
        for i in range(len(data)):
            comment = json.loads(data[i])
            feats[i] = extract1(comment["body"])
            feats[i][-1] = cat_to_i[comment["cat"]]
            if feats[i][-1] == 0:
                feats[i][29:173] = left_data[left_id[comment["id"]]]
            elif feats[i][-1] == 1:
                feats[i][29:173] = center_data[center_id[comment["id"]]]
            elif feats[i][-1] == 2:
                feats[i][29:173] = right_data[right_id[comment["id"]]]
            elif feats[i][-1] == 3:
                feats[i][29:173] = alt_data[alt_id[comment["id"]]]
       
    np.savez_compressed( args.output, feats)
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

