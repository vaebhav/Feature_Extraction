#########################
## author - Vaebhav
#########################

import re
from collections.abc import Iterable
import numpy as np
import collections
import nltk
from pprint import pprint
import itertools as it
from nltk.corpus import stopwords
import pandas as pd
import string

def unique_tokens(token_list,freqDict=False):

    '''

    Accepts inputs as list with indiviual tokens (tokenized vocabulary) as list indexes.
    Returns a dictionary of unique tokens/vocabulary words across the doucments.

    Keyword Arguments:
    freqDict ---> Default FALSE , if TRUE returns the occurence of each token across the document sets. Should be true in case of TFidfVector

    '''

    #print("tokenList ---->",token_list)

    # if len(token_list[0]) > 1:
    #     wordTokenFlag = True
    # else:
    #     wordTokenFlag = False
    #
    # if wordTokenFlag:
    #     token_list = [sent if isinstance(sent,tuple) else sent.split() for sent in token_list ]


    #print("Flag-------->",wordTokenFlag)
    #print("tokenList ---->",token_list)

    unq_token = collections.defaultdict()
    freq_hash = collections.defaultdict()
    unq_token.default_factory = unq_token.__len__
    punct_list = list(map(lambda x: x *2, string.punctuation))
    punct_list.extend(list(string.punctuation))
    #print('Punc List ----',punct_list)

    for input_token in range(len(token_list)):
        #unq_token , freq_hash = generate_unq_tokens(token_list[input_token])
        key = str()
        if isinstance(token_list[input_token],(list,np.ndarray)):
            for each in range(len(token_list[input_token])):
                try:
                    key = token_list[input_token][each].lower().strip()
                    if len(key) > 1 and key not in punct_list:
                        unq_token[key]
                        #print('Key ---',key,'Len--',len(key))
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
                key = token_list[input_token]
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

          #print('Doc Size ----',doc_size)

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
