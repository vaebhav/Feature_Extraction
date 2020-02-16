#!/usr/local/bin/python3

#########################
## author - Vaebhav
#########################

from sys import getsizeof
import re
from collections.abc import Iterable
import numpy as np
from collections import defaultdict
from collections import Counter
from collections import OrderedDict
from pprint import pprint
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix,lil_matrix
import pandas as pd
import string
import pickle
#from sklearn.externals import joblib
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from preprocessor.text_preprocessor import TextPreprocessor
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool,Manager,RawArray
from functools import partial
from multiprocessing import cpu_count
from numba import double, jit
from memory_profiler import profile
# from multiprocessing import Manager
# from multiprocessing import RawArray
#import array
#from pickle4reducer import Pickle4Reducer

var_dict = {}

def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    print('\t\tinside worker')

    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

class TextExtraction:

    def list2str(self,file_content):

        if isinstance(file_content[0],str):
            return ' '.join([word.strip() for word in file_content])
        elif isinstance(file_content[0],(list,Iterable)):
            return ' '.join([token.strip() for word in file_content for token in word])

    def _word_ngrams(self,tokens, stop_words=None):

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


    def tokenizeDocument(self,doc,tokenize='sentence',returnDoc=False):


        if tokenize == 'sentence':
            spt = nltk.PunktSentenceTokenizer()

            if isinstance(doc,(pd.DataFrame,pd.Series)):
                tokens = doc.apply(lambda row : spt.tokenize(' '.join(row))).values
            elif isinstance(doc,list):
                tokens = list()
                for each in doc:
                    tokens.append(spt.tokenize(each))
            else:
                tokens = spt.tokenize(doc)
        elif tokenize == 'word':

            wpt = nltk.WordPunctTokenizer()

            if isinstance(doc,(pd.DataFrame,pd.Series)):
                tokens = doc.apply(lambda row: wpt.tokenize(' '.join(row))).values
            elif isinstance(doc,list):
                tokens = list()
                for each in doc:
                    tokens.append(wpt.tokenize(each))
            else:
                tokens = wpt.tokenize(doc)

        preProcObj = TextPreprocessor()

        #final_tokens = preProcObj.preprocess_text(tokens,[preProcObj.removeStopWords,preProcObj.removeNumbers,preProcObj.removeEmptyString],strFlag=False)
        #final_tokens = preProcObj.preprocess_text(tokens,[preProcObj.lowercase,preProcObj.lemmatize,preProcObj.removePunctuation,preProcObj.removeEmptyString,preProcObj.removehypen],strFlag=False)
        final_tokens = preProcObj.preprocess_text(tokens,[preProcObj.lowercase],strFlag=False)

        if returnDoc:
        # re-create document from filtered tokens
            doc = ' '.join(final_tokens)
            return final_tokens,doc
        else:
            return final_tokens

    def unique_tokens(self,token_list,freqDict=False,token_filter=None):

        '''
        Accepts inputs as list with indiviual tokens (tokenized vocabulary) as list indexes.
        Returns a dictionary of unique tokens/vocabulary words across the doucments.

        Keyword Arguments:
        freqDict ---> Default FALSE , if TRUE returns the occurence of each token across the document sets. Should be true in case of TFidfVector

        '''
        total_freq_sum = 0
        unq_token = defaultdict()
        #unq_token = OrderedDict()
        freq_hash = defaultdict()

        unq_token.default_factory = unq_token.__len__

        #print('--->',unq_token)
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
                                total_freq_sum +=1
                            else:
                                freq_hash[key] = 1
                                total_freq_sum +=1
                        else:
                            continue
                    except KeyError:
                        continue
            else:
                try:
                    key = token_list[input_token].strip()
                    if len(key) > 1 and key not in punct_list:
                        unq_token[key]
                        if key in freq_hash:
                            freq_hash[key] += 1
                            total_freq_sum +=1
                        else:
                            freq_hash[key] = 1
                            total_freq_sum +=1
                    else:
                        continue
                except KeyError:
                    continue


        if token_filter is not None:
            filtered_tokens = defaultdict()
            filtered_tokens.default_factory = filtered_tokens.__len__
            for each in freq_hash:
                token_freq = freq_hash[each]
                percent_freq = (token_freq / total_freq_sum) * 100

                if percent_freq <= token_filter:
                    # if each == 'ny':
                    #     print(each,'\t\t',percent_freq,'\t\t',total_freq_sum)
                    filtered_tokens[each]
                    unq_token.pop(each,None)
                else:
                    continue

        if freqDict:
            return unq_token,freq_hash
        else:
            return unq_token

    def generate_unq_tokens_singleDoc(self,input_document):

        unq_token = defaultdict()
        freq_hash = defaultdict()
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
        if token_filter is not None:
            for each in freq_hash:
                token_freq = freq_hash[each]
                percent_freq = (token_freq / total_freq_sum) * 100
                if percent_freq <= token_filter:
                    unq_token.pop(each,None)
                else:
                    continue

        return unq_token,freq_hash

class CountVector(TextExtraction):

    def count_freq(self,arr):
        return Counter(map(lambda x:x.lower() ,arr))

    def CountVector(self,input_docs,returnDict=False):
        '''
        Accepts inputs as list with indiviual documents as list indexes.
        Returns a data frame with the Frequency of unique tokens/words across individual documents.

        Keyword Arguments:
        returnDict ---> Default FALSE , if TRUE returns the dictionary rather then dataframe.

        Exceptions Raised:
        TypeError ---> In case of the Input Document is Iterable or not.

        '''

        CountHash = defaultdict()
        feature_set = unique_tokens(input_docs)
        temp_dict_freq = Counter()

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


class TFIDFVector(TextExtraction):

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


class Leaders(TextExtraction):

    def calcEuclideanDistance_matrix(self,in_row,main_matrix):

        dist = (in_row - main_matrix)**2

        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)#,dtype=np.int8)
        dist = list(np.where(dist==0)[0])

        return dist

    def calcJensenshannonDivergence(self,in_row1,in_row2):

        return jensenshannon(in_row1,in_row2)

    def calcEuclideanDistance(self,in_row1,in_row2,ref_hash=False):

        if ref_hash:
            in_row_idx1 = [ref_hash[x] for x in in_row1]
            in_row_idx2 = [ref_hash[x] for x in in_row2]
            euclidean_dist = euclidean(in_row_idx1,in_row_idx2)
        else:
            euclidean_dist = euclidean(in_row1,in_row2)

        return euclidean_dist

    def createClusterSpace_test2(self,doc,dataframe=False,bagCount=0):

        bow_obj = BagOfWords()

        empty_hash_flag = True

        vector_matrix = bow_obj.createVectorSpace(doc,prob_dist=False)

        if isinstance(vector_matrix,tuple):
            dist_matrix = np.memmap(dist_path_shape[0], dtype=np.int8,mode='r',shape=vector_matrix[1])
        else:
            dist_matrix = vector_matrix
            del vector_matrix

        print('--------------- cluster --------------------')

        clusterHash = defaultdict(list)

        dist_matrix_shape = dist_matrix.shape

        cpu = cpu_count()
        pool = Pool(cpu - 2)

        map_asyncresults = list()
        
        pool_result = pool.map_async(partial(self.calcEuclideanDistance_matrix,main_matrix=dist_matrix),dist_matrix,callback=map_asyncresults.append)

        pool.close()
        pool.join()

        clusterHash.fromkeys([tuple(x) for x in pool_result.get()])
       
        if dataframe:
            # df = pd.DataFrame.from_dict({ k: dict(v) for k,v in vectorSpace.items() },
            #                     orient="index",columns=TokenHash.keys())
            #df = db.from_sequence(vectorSpace).to_dataframe()

            df = pd.DataFrame(values,columns=dfColumns.keys())

            return df
        else:
            return clusterHash

    def createClusterSpace(self,doc,dataframe=False,bagCount=0):

        doc_len = len(doc)

        dfColumns = OrderedDict()

        if bagCount > 1:
            ngram_doc = self.generate_ngrams(doc,bagCount)
            TokenHash = self.unique_tokens(ngram_doc,freqDict=True)
        else:
            word_doc = self.list2str(doc).split()
            TokenHash = self.unique_tokens(word_doc)#,token_filter=0.7)
            

        lenDoc_unq = len(TokenHash)

        if len(TokenHash) == 0:
            raise ValueError("After token filtering, no terms remain. Try a lower"
                             " token_filter value")
        empty_hash_flag = True

        no_of_leaders = 1

        clusterHash = defaultdict(list)

        for doc_idx,token in enumerate(doc):
            if bagCount > 1:
                ngram = self.generate_ngrams(token,bagCount)
            else:
                if isinstance(token,str):
                    ngram = token.split()
                    indptr_len = len(ngram)
                elif isinstance(token,list):
                    ngram = token
                    indptr_len = len(token)

            if len(token) != 0:
                if empty_hash_flag == True:
                    if bool(clusterHash) == False:
                        clusterHash[no_of_leaders].append((token,doc_idx))
                        empty_hash_flag = False
                else:
                    for cluster in clusterHash:
                        max_euc_dist = 0
                        for data_row1,data_row2 in zip(clusterHash[cluster][0],[token]):
                            euclidean_distance = self.calcEuclideanDistance(data_row1,data_row2,TokenHash)
                            if max_euc_dist <= euclidean_distance:
                                max_euc_dist = euclidean_distance

                    if max_euc_dist == 0:
                        no_of_leaders =+ 1
                        clusterHash[no_of_leaders].append((token,doc_idx))
                    else:
                        clusterHash[no_of_leaders].append((token,doc_idx))

        if dataframe:
            # df = pd.DataFrame.from_dict({ k: dict(v) for k,v in vectorSpace.items() },
            #                     orient="index",columns=TokenHash.keys())
            #df = db.from_sequence(vectorSpace).to_dataframe()

            df = pd.DataFrame(values,columns=dfColumns.keys())

            return df
        else:
            return clusterHash

class BagOfWords(TextExtraction):

    def __init__(self):
        hd = os.path.expanduser('~')
        try:
            if not nltk.data.find(hd + '/nltk_data/tokenizers/punkt'):
                nltk.download('punkt')
        except LookupError:
            nltk.download('punkt')

    def keyCheck(self,key,hash):
        if key not in hash:
            yield key

    def createVectorSpace(self,doc,dataframe=False,bagCount=0,prob_dist=False):

        doc_len = len(doc)

        dfColumns = OrderedDict()

        if bagCount > 1:
            ngram_doc = self.generate_ngrams(doc,bagCount)
            TokenHash = self.unique_tokens(ngram_doc,freqDict=False)
        else:
            word_doc = self.list2str(doc).split()
            TokenHash = self.unique_tokens(word_doc)#,token_filter=0.7)

        lenDoc_unq = len(TokenHash)

        #pprint(TokenHash)

        if len(TokenHash) == 0:
            raise ValueError("After token filtering, no terms remain. Try a lower"
                             " token_filter value")
        #set_keys = OrderedSet(TokenHash.keys())

        if prob_dist == True:
            #values = np.zeros((doc_len,lenDoc_unq),dtype=np.float32)
            values = csr_matrix((doc_len,lenDoc_unq),dtype=np.float32)
        else:
            #values = np.zeros((doc_len,lenDoc_unq),dtype=np.int8)
            try:
                values = np.zeros((doc_len,lenDoc_unq),dtype=np.int8)
            except MemoryError:
                values = np.memmap('memmapped.dat', dtype=np.int8,mode='w+', shape=(doc_len,lenDoc_unq))

        for doc_idx,token in enumerate(doc):

            if bagCount > 1:
                ngram = self.generate_ngrams(token,bagCount)
            else:
                if isinstance(token,str):
                    ngram = token.split()
                    indptr_len = len(ngram)
                elif isinstance(token,list):
                    ngram = token
                    indptr_len = len(token)

            counterHash = defaultdict()
            indices = list()
            indptr = list()

            #### adding zero to avoid ValueError -----> index pointer should start with 0
            indptr.append(0)

            for word in ngram:

                if len(word) != 1:
                    if word in TokenHash:
                        word_idx = TokenHash[word]
                        if word_idx not in counterHash:
                            dfColumns[word] = word_idx
                            counterHash[word_idx] = 1
                        else:
                            counterHash[word_idx] += 1
                    else:
                        continue
            #print(doc_idx,'\t\t\t',counterHash)
            ##### Another way , For implmenting BAG of WORDS Method #######
            #diff_keys = list(set_keys.difference(counterHash.keys()))
            #diff_keys = list(filter(lambda x: self.keyCheck(x,counterHash),TokenHash.keys()))
            #diff_dict = dict.fromkeys(diff_keys,0)

            #counterHash.update(diff_dict)
            #counterHash = OrderedDict(counterHash.items())

            indices = list(counterHash.keys())

            if prob_dist == True:
                row_value = [float(x/sum(x)) for x in list(counterHash.values())]
            else:
                row_value = list(counterHash.values())
            indptr.append(len(indices))

            values[doc_idx] = csr_matrix((row_value,indices,indptr),shape=(len(indptr) - 1, lenDoc_unq)).toarray()#,dtype=np.int8)

        if dataframe:
            # df = pd.DataFrame.from_dict({ k: dict(v) for k,v in vectorSpace.items() },
            #                     orient="index",columns=TokenHash.keys())
            #df = db.from_sequence(vectorSpace).to_dataframe()

            df = pd.DataFrame(values,columns=dfColumns.keys())
            
            return df
        else:
            if isinstance(values,np.memmap):
                matrix_filename = values.filename
                matrix_shape = values.shape

                del values

                return (matrix_filename,matrix_shape)
            else:
                return values
