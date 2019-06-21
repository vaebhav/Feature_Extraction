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
from Preprocessor_Initial import Preprocessor



class BagOfWords:
    def __init__(self):
        if not nltk.data.find('tokenizers/punkt'):
            nltk.download('punkt')

    def list2str(self,file_content):

        return ' '.join([word.strip() for word in file_content])

    def _word_ngrams(self, tokens, stop_words=None):

        """
        Turn tokens into a sequence of n-grams after stop words filtering
        """

        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = (1,1)
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
        print("---IN----",input_list)
        return zip(input_list, input_list[1:])
        #return zip(nltk.word_tokenize(self.list2str(input_list)), nltk.word_tokenize(self.list2str(input_list[1:])))

    def generate_ngrams(self,input_list, n):

        '''
        Generate N Grams for an input list
        Arguments:
        input_list ---> Input list for which N - gram needs to be generated.
        n ---> Specifies the range for which the grams are to be generated
        '''

        return zip(*[input_list[i:] for i in range(n)])

    def gen_ngrams_scikit_learn(self,input_tokens,min_n=1,max_n=1):
        original_tokens = []
        tokens = input_tokens

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

    def createVectorSpace(self,doc):
        print("Ehllo")

    def tokenizeDocument(self,doc,tokenize='sentence'):

        if tokenize == 'sentence':
            spt = nltk.PunktSentenceTokenizer()
            tokens = spt.tokenize(doc)
        else:
            wpt = nltk.WordPunctTokenizer()
            tokens = wpt.tokenize(doc)

        preProcObj = Preprocessor()

        filtered_tokens = preProcObj.preprocess_text(tokens,[preProcObj.removeStopWords],False)

        final_tokens = preProcObj.preprocess_text(tokens,[preProcObj.removeNumbers,preProcObj.removePunctuation,preProcObj.removeStopWords,preProcObj.removeNumbers,preProcObj.lowercase],False)
        # re-create document from filtered tokens
        doc = ' '.join(filtered_tokens)
        #return filtered_tokens,doc
        return final_tokens,doc


Testcorpus = """
The Little Pine Tree.
A little pine tree was in the woods.
It had no leaves. It had needles.
The little tree said, "I do not like needles. All the other trees in the woods have pretty leaves. I want leaves, too. But I will have better leaves. I want gold leaves."
Night came and the little tree went to sleep. A fairy came by and gave it gold leaves.
When the little tree woke it had leaves of gold.
It said, "Oh, I am so pretty! No other tree has gold leaves."
Night came.
A man came by with a bag. He saw the gold leaves. He took them all and put them into his bag.
The poor little tree cried, "I do not want gold leaves again. I will have glass leaves."
So the little tree went to sleep. The fairy came by and put the glass leaves on it.
The little tree woke and saw its glass leaves.
How pretty they looked in the sunshine! 'No other tree was so bright.
Then a wind came up. It blew and blew.
The glass leaves all fell from the tree and were broken.
Again the little tree had no leaves. It was very sad, and said, "I will not have gold leaves and I will not have glass leaves. I want green leaves. I want to be like the other trees."
And the little tree went to sleep. When it woke, it was like other trees. It had green leaves.
A goat came by. He saw the green leaves on the little tree. The goat was hungry and he ate all the leaves.
Then the little tree said, "I do not want any leaves. I will not have green leaves, nor glass leaves, nor gold leaves. I like my
needles best."
And the little tree went to sleep. The fairy gave it what it wanted.
When it woke, it had its needles again. Then the little pine tree was happy.
"""

bowObj = BagOfWords()

token_sent,temp = bowObj.tokenizeDocument(Testcorpus)

for each in bowObj.gen_bigrams(token_sent):
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<---->",each)
