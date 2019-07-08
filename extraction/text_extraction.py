#!/usr/local/bin/python3

#########################
## author - Vaebhav
#########################

import re
from collections.abc import Iterable
import numpy as np
from collections import defaultdict
from pprint import pprint
from nltk.corpus import stopwords
import pandas as pd
import string
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from preprocessor.text_preprocessor import TextPreprocessor

def unique_tokens(token_list,freqDict=False):

    '''

    Accepts inputs as list with indiviual tokens (tokenized vocabulary) as list indexes.
    Returns a dictionary of unique tokens/vocabulary words across the doucments.

    Keyword Arguments:
    freqDict ---> Default FALSE , if TRUE returns the occurence of each token across the document sets. Should be true in case of TFidfVector

    '''

    unq_token = defaultdict()
    freq_hash = defaultdict()
    unq_token.default_factory = unq_token.__len__
    punct_list = list(map(lambda x: x *2, string.punctuation))
    punct_list.extend(list(string.punctuation))

    for input_token in range(len(token_list)):
        key = str()
        if isinstance(token_list[input_token],(list,np.ndarray)):
            for each in range(len(token_list[input_token])):
                try:
                    key = token_list[input_token][each].lower().strip()
                    if len(key) > 1 and key not in punct_list:
                        unq_token[key]
                        if key in freq_hash:
                            freq_hash[key] += 1
                        else:
                            freq_hash[key] = 1
                    else:
                        next
                except KeyError:
                    next
        else:
            try:
                key = token_list[input_token].strip()
                if len(key) > 1:
                    unq_token[key]
                    if key in freq_hash:
                        freq_hash[key] += 1
                    else:
                        freq_hash[key] = 1
                else:
                    next
            except KeyError:
                next


    if freqDict:
        return unq_token,freq_hash
    else:
        return unq_token

def generate_unq_tokens_singleDoc(input_document):

    unq_token = collections.defaultdict()
    freq_hash = collections.defaultdict()
    unq_token.default_factory = unq_token.__len__

    for doc in input_document:
        key = str()
        if isinstance(doc,(list,np.ndarray)):
            for each in range(len(doc)):
                try:
                    key = doc[each].lower().strip()
                    if len(key) > 1:
                        unq_token[key]
                        if len(key) > 1:
                            if key in freq_hash:
                                freq_hash[key] += 1
                            else:
                                freq_hash[key] = 1
                    else:
                        next
                except KeyError:
                    next
        else:
            try:
                key = doc.lower().strip()
                if len(key) > 1:
                    unq_token[key]
                    if len(key) > 1:
                        if key in freq_hash:
                            freq_hash[key] += 1
                        else:
                            freq_hash[key] = 1
                else:
                    next
            except KeyError:
                next

    return unq_token,freq_hash



class CountVector:

    def count_freq(self,arr):
        return collections.Counter(map(lambda x:x.lower() ,arr))

    def CountVector(self,input_docs,returnDict=False):
        '''
        Accepts inputs as list with indiviual documents as list indexes.
        Returns a data frame with the Frequency of unique tokens/words across individual documents.

        Keyword Arguments:
        returnDict ---> Default FALSE , if TRUE returns the dictionary rather then dataframe.

        Exceptions Raised:
        TypeError ---> In case of the Input Document is Iterable or not.

        '''

        CountHash = collections.defaultdict()
        feature_set = unique_tokens(input_docs)
        temp_dict_freq = collections.Counter()

        if isinstance(input_docs,str) or not isinstance(input_docs, Iterable):
            raise TypeError('Input Document/Corpus tokens is not Iterable')


        bool_list_flag = any(isinstance(el, list) for el in input_docs)

        if bool_list_flag:

            for inp in input_docs:
                temp_dict_freq = self.count_freq(inp)
                for tokens in feature_set:
                    if tokens in temp_dict_freq:
                        next
                    else:
                        temp_dict_freq.update({tokens:0})
                CountHash[input_docs.index(inp)] = temp_dict_freq
        else:
            temp_dict_freq = self.count_freq(input_docs)
            for tokens in feature_set:
                if tokens in temp_dict_freq:
                    next
                else:
                    temp_dict_freq.update({tokens:0})
            CountHash[0] = temp_dict_freq


        df = pd.DataFrame.from_dict(CountHash).T

        if returnDict:
            return df,CountHash
        else:
            return df


class TFIDFVector:

      def count_freq(self,arr):
          return collections.Counter(map(lambda x:x.lower() ,arr))

      def computeTFscore(self,vocabulary_freq,doc_len):

          '''
          Computes TF score for each token/word passed.

          vocabulary_freq ---> Frequency of token/word/vocab in the document
          doc_len ---> Length/Size of the Document under which the token exists

          '''

          bowCount = doc_len
          tfscore = vocabulary_freq/float(bowCount)

          return tfscore

      def computeIDFscore(self,num_doc,token_docfreq):

          '''
          Computes IDF score for each token/word passed.

          num_doc ---> No of documents.
          token_docfreq ---> Total frequency of token/word/vocab across all the documents

          '''

          idfscore = np.log( num_doc / token_docfreq + 1)

          return idfscore




      def TFidfVector(self,input_docs,returnDict=False):
          '''
          Accepts inputs as list with indiviual documents as list indexes.
          Returns a data frame with the TF-IDF score across each document.

          Arguments:
          returnDict ---> Default FALSE , if TRUE returns the dictionary rather then dataframe.


          '''

          bool_list_flag = any(isinstance(el, list) for el in input_docs)

          #bool_list_flag = False

          if not bool_list_flag:
              try:
                  vocabulary_set,vocabulary_set_docfreq = generate_unq_tokens_singleDoc(input_docs)
                  doc_size = len(input_docs[0])
              except ValueError:
                  print("Unique tokens were not fetched for further computation")
                  raise Exception ("TF-IDF Vector cannot be created using a single input document")
          else:
              try:
                  vocabulary_set,vocabulary_set_docfreq = unique_tokens(input_docs,freqDict=True)
                  doc_size = len(input_docs)
              except ValueError:
                  print("Unique tokens were not fetched for further computation")

          tfidfHash = collections.defaultdict()

          temp_dict_freq = collections.Counter()

          if isinstance(input_docs,str) or not isinstance(input_docs, Iterable):
              raise TypeError('Input Document/Corpus tokens is not Iterable')


          for count,inp in enumerate(input_docs):
              temp_dict_freq = self.count_freq(inp)
              imdHash = collections.defaultdict()
              for token in vocabulary_set:
                  if token in temp_dict_freq:
                      tfscore = self.computeTFscore(temp_dict_freq[token],len(inp))
                      idfscore = self.computeIDFscore(doc_size,vocabulary_set_docfreq[token])
                      imdHash.update({
                                  token: {'tf-score': tfscore,
                                         'idf-score': idfscore,
                                         'tf-idf': tfscore * idfscore
                                     }
                                   })
                      #imdHash.update({token:tfscore * idfscore})
                  else:
                      imdHash.update({
                                  token: {'tf-score': 0,
                                         'idf-score': 0,
                                         'tf-idf': 0
                                     }
                                   })
                      #imdHash.update({token:0})
              tfidfHash[count] = imdHash

          df = pd.DataFrame.from_dict(tfidfHash).T

          if returnDict:
              return df,tfidfHash
          else:
              return df


class BagOfWords:

    def __init__(self):
        hd = os.path.expanduser('~')
        try:
            if not nltk.data.find(hd + '/nltk_data/tokenizers/punkt'):
                nltk.download('punkt')
        except LookupError:
            nltk.download('punkt')

    def list2str(self,file_content):

        return ' '.join([word.strip() for word in file_content])

    def _word_ngrams(self, tokens, stop_words=None):

        """
        Turn tokens into a sequence of n-grams after stop words filtering
        """
        tokens = self.list2str(tokens).split()

        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = (1,2)
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                #tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def generate_bigrams_traditional(self,input_list):

        '''
        Generate BiGrams for an input list . Traditional way
        Arguments:
        input_list ---> Input list for which N - gram needs to be generated.
        '''
        input_list = self.list2str(input_list).split()

        bigram_list = []
        for i in range(len(input_list)-1):
            bigram_list.append((input_list[i], input_list[i+1]))
        return bigram_list

    def gen_bigrams(self,input_list):

        '''
        Generate BiGrams for an input list
        Arguments:
        input_list ---> Input list for which N - gram needs to be generated.
        '''


        input_list = self.list2str(input_list).split()

        result = zip(*(input_list, input_list[1:]))

        return [' '.join(n) for n in result]

    def generate_ngrams(self,input_list, n):

        '''
        Generate N Grams for an input list
        Arguments:
        input_list ---> Input list for which N - gram needs to be generated.
        n ---> Specifies the range for which the grams are to be generated
        '''
        if not isinstance(input_list,str):
            input_list = self.list2str(input_list).split()
        elif isinstance(input_list,str):
            input_list = input_list.split()


        result = zip(*[input_list[i:] for i in range(n)])

        return [ ' '.join(i) for i in result ]

    def gen_ngrams_scikit_learn(self,input_tokens,min_n=1,max_n=1):
        original_tokens = []

        tokens = tokens = self.list2str(input_tokens).split()

        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

        n_original_tokens = len(original_tokens)
        tokens_append = tokens.append
        space_join = " ".join

        for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def count_freq(self,arr):
        return collections.Counter(map(lambda x:x.lower() ,arr))

    def createVectorSpace(self,doc,bagCount=0):

        lenDoc = len(doc)

        if bagCount > 1:
            ngram_doc = self.generate_ngrams(doc,bagCount)
            TokenHash = unique_tokens(ngram_doc,freqDict=False)
        else:
            word_doc = self.list2str(doc).split()
            TokenHash = unique_tokens(word_doc)

        vectorSpace = defaultdict()

        for idx,sent_token in enumerate(doc):
            if bagCount > 1:
                ngram = self.generate_ngrams(sent_token,bagCount)
            else:
                ngram = sent_token.split()
            for word in ngram:
                if word in TokenHash and word not in vectorSpace:
                #if word in TokenHash:
                    #if word not in vectorSpace:
                        vectorSpace[word] = np.zeros(shape=lenDoc,dtype=int)
                        vectorSpace[word][idx] = 1
                else:
                    vectorSpace[word][idx] = 1

        df = pd.DataFrame.from_dict(vectorSpace)

        return df


    def tokenizeDocument(self,doc,tokenize='sentence'):

        if tokenize == 'sentence':
            spt = nltk.PunktSentenceTokenizer()
            tokens = spt.tokenize(doc)
        else:
            wpt = nltk.WordPunctTokenizer()
            tokens = wpt.tokenize(doc)

        preProcObj = TextPreprocessor()

        final_tokens = preProcObj.preprocess_text(tokens,[preProcObj.removeStopWords,preProcObj.removeNumbers],strFlag=False)

        # re-create document from filtered tokens
        doc = ' '.join(final_tokens)

        return final_tokens,doc
