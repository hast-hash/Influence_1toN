# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:14:43 2022

@author: hast
"""
# Influence 1toN: Query search from files (text reuse)
# This codes investigate the influence of one seminal text (a source text) on later texts (target texts) 
# - checking keywords if they are appeared in a collection of texts,
#   using traditional tf-idf and cosine similarity comparison (comparing exact the same words only)
#input: source text (a vocabulary list you want to look for in target texts)
#       target texts (a collection of texts in a folder)
#output:a text file which includes a list of target file name and found keywords    

#loading libraries
import glob
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk 
from nltk.stem import SnowballStemmer

#Specify the following variables before you implement this program
#the filename of the source text
source_text_file_name = 'Source_text.txt'
#the folder name which contains the target texts
target_texts_folder_name = 'Target_texts'
#the header name for saving output files/a file
outfile_header = 'cos_sim'
#omitting words from the source text
exempt_source = []
#example:
#exempt_source = ['al']
#letters that divide folders in your environment. '\\' or '/'
slash_char = '\\'
#encoding type for all the texts
targetdataencoding='utf-8'

#variables the codes uses
#the folder name which contains data and pushing results 
file_directory = os.getcwd()
#the direct filename for source
text_source = file_directory+slash_char+source_text_file_name
#the direct foldername for target
dir_data = file_directory+slash_char+target_texts_folder_name
nostr = ""
text = ""

#important variables
#a list of all the source and target text = text a + textb
alltexts = []
#the source text(word list)
texta = []
#a list of target texts(text themselves)
textb = []
#a list of target filenames 
filenames = []
#the tfidf table (numeric only, terms x target)
tfidfs = []
#a list of terms (terms themselves)
terms = []
#cosine similarity table ((source + target) x (source + target))
sim = []

#preprocessing
#1.cutting carriage returns
#2.cutting blank lines


def preprocessing(text = text):
    #changing a dividing character into a space
    text = re.sub('\n|—', ' ', text)
    #erasing note numbers [nnn]
    text = re.sub('\[[0-9]{1,}\]', ' ', text)
    #erasing blank lines
    text = re.sub('^..$|^.$', '', text)
    #erasing unnecessary spaces
    text = re.sub('  ', ' ', text)

    #lowering all the text
    text = text.lower()
    #stemming
    text = re.findall(r'\"\[.+\]\"+|\'\[.+\]\'+|[\w\S]+|\w+|\S\w*', text)
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word) for word in text])

    return text

#loading target texts into variables, dir_data specifies the folder name
def load_target_texts(dir_data=dir_data):
    #getting all filenames in the target folder
    files = glob.glob(dir_data+slash_char+"*")
    text2 = []
    t = []
    text = ""
    filenames = []
    texts = []
    #opening each file and getting atrget texts
    for filename in files:
        with open(filename, "r", encoding=targetdataencoding) as f:
            text = f.read()
            text = preprocessing(text)
            idx = len(dir_data+slash_char)
            filename = filename[idx:]
            #setting filenames and texts
            t = [filename, text]
            text2.append(t)
            filenames.append(filename)
            texts.append(text)
    #text2 = [filenames, text], filenames = [filenames], texts = [texts]
    return text2, filenames, texts

#loading the source text into variable 
#source text should be a list of words you comapre
def load_source_text(text_source = text_source, exempt_source = exempt_source):
    with open(text_source, "r", encoding=targetdataencoding) as f:
        text = f.read()
        #preprocessing
        text0 = text
        j = ""
        for i in text0.split("\n"):
            j = j + "\""+i+"\""+"\n"
        text = preprocessing(j)
#        text = preprocessing(text)
        #erasing omit words if any
        #omitting words that start #
        text = re.sub('\#.*? ', '', text)
        text = re.sub(' \#.*?$', '', text)        
        if len(exempt_source):
            for i in exempt_source:
#                i = preprocessing(i)
                j = '^\"'+i+' |[ \"]'+i+'[ \""]| '+i+'\"'
                text = re.sub(j, ' ', text)
        #dictionary for original words
        sdic = {}
        for i in text0.split("\n"):
#            i = "\""+i+"\""
            j = preprocessing(i)
            if sdic.get(j) is None:
                k = [i]
            else:
                k = sdic[j]
                k.append(i)
            sdic[j] = k
    return [text], sdic

#calculating tfidf using TfidfVectorizer (sklearn)
#alltexts includes a source text and target texts
def get_tfidf(alltexts = alltexts):
    #texts are lowercased and stopwords are erased. nomrization is calculated without any adjustment(smooth_idf is off) 
    vectorizer = TfidfVectorizer(lowercase=True, norm='l2', smooth_idf=False, stop_words='english')
    #calculating
    vecs = vectorizer.fit_transform(alltexts)
    #getting tfidfs data and transforming it to an array 
    #         terms1           terms2       terms3 ... all the terms in the whole collection
    #target1  value of tfidfs 
    #target2
    #target3
    #...
    tfidfs = vecs.toarray()
    #getting a list of terms (all the terms appeared in the whole text collection)
    terms = vectorizer.get_feature_names()
    return tfidfs, terms

#loading target texts and the source text. Putting them into alltexts
text2, filenames, textb = load_target_texts(dir_data)
texta, sdic = load_source_text(text_source)
alltexts = texta
alltexts.extend(textb)

#implementing tfidf and cosine similarity
tfidfs, terms = get_tfidf(alltexts)

#calculating cos/sim using cosine_similarity (sklearn)
#        target1           target2       target3 ...
#target1 value of cos/sim
#target2
#target3
#...
sim = cosine_similarity(tfidfs)

#heatmap for debugging
def draw_heatmap(sim=sim):
    plt.figure()
    plt.figure(figsize=(12, 8))
    sns.heatmap(sim, cmap='Blues')

#saving variables in list type
def list_save(filename, list, columns=nostr, index=nostr):
    if columns == "" and index == "":
        df = pd.DataFrame(list)
    elif columns == "" and index != "":
        df = pd.DataFrame(list, index=index)
    else:
        df = pd.DataFrame(list, columns=columns, index=index)
    df.to_csv(filename, encoding=targetdataencoding)

list_ext = ".csv"
sf = file_directory+slash_char+outfile_header
#examples of saving variables
#list_save(sf+"_result_sim"+list_ext, sim)
#list_save(sf+"_result_tfidfs"+list_ext, tfidfs)
#list_save(sf+"_result_terms"+list_ext, terms)
#list_save(sf+"_result_text2"+list_ext, text2)
#list_save(sf+"_result_filenames"+list_ext, filenames)

#displaying the same keywords
#file number, filename, value of cos/sim, a list of shared keywords 
def sim_out(sim = sim, tfidfs = tfidfs, terms = terms, filenames = filenames):
    #x contains similarity of source and targets 
    x = sim[0]
    j = 0
    sim_out = []
    #a contains source tfidfs
    a = tfidfs[0]
    #checking each term of tfidf in source by targets
    #if similarity is 0, it skips
    for i in x:
        terms_num = []
        terms_list = []
        if i != 0 and j != 0:
            l = 0
            #getting tfidfs in one target
            b = tfidfs[j]
            #calculating the source times the target, i.e. if a word is shared, it shows above 0.
            for k in a*b:
                #when the codes finds a shared word, terms_list includes the keyword itself.
                if k != 0:
                    terms_num.append(l)
                    if sdic.get(terms[l]) is None:
                        terms_list.append(terms[l])
                    else:
                        for m in sdic[terms[l]]:
                            terms_list.append(m)
                l += 1
            
            #preparing data for displaying/savaing
            #num: file number
            #work: title
            #len_words: length of texts
            #val_sim: values of cos\sim
            #terms: term list (without comma)
            #file number
            num = j
            #title, cutting unnecessary information
            work = filenames[j-1]
            work = re.sub('^.*_|.txt', '', work)  
            #length of texts
            len_words_tmp = alltexts[num].split(' ')
            k = 0
            for m in len_words_tmp:
                if m == "":
                    len_words_tmp.pop(k)
                k += 1
            len_words = len(len_words_tmp)
            val_sim = i
            #term list (without comma)
            terms_out = ""
            for m in terms_list:
                terms_out = terms_out + m + " "
            
            #putting variables into a list
            sim_out.append([num, work, len_words, val_sim, terms_out])
        j += 1
    #[file number, filename, value of cos/sim, a list of shared keywords]
    return sim_out



#getting the query search result(target texts chich includes source keywords) for display
sim_out = sim_out(sim, tfidfs, terms, filenames)
#sim_out2 = sim_out2(sim_out)
#printout the result on the screen and save it as a file
print(pd.DataFrame(sim_out))
list_save(sf+"_result_sim_out"+list_ext, sim_out)

#Other ways of calculating cosine similarity

#vectorizer2 = TfidfVectorizer(lowercase=True, norm='l2', smooth_idf=False, stop_words='english')
#vecs2 = vectorizer2.fit_transform(textb)
#tfidfs2 = vecs2.toarray()
#terms2 = vectorizer2.get_feature_names()
#source_tfidf = vectorizer2.transform(texta)
#sim2 = cosine_similarity(source_tfidf, vecs2)[0]

#from sklearn.metrics.pairwise import linear_kernel
#vectorizer3 = TfidfVectorizer(lowercase=True, norm='l2', smooth_idf=False, stop_words='english')
#vecs3 = vectorizer3.fit_transform(alltexts)
#sim3 = linear_kernel(vecs3[0:1], vecs3).flatten()

