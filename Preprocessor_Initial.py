import re
from collections.abc import Iterable
import numpy as np
import collections
import nltk
from pprint import pprint
import itertools as it
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
#from Feature_Extraction import CountVector,unique_tokens
import sys

sys.path.append('C:\\NLP_Practice')


# Some strings for ctype-style character classification
whitespace = ' \t\n\r\v\f'
ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ascii_letters = ascii_lowercase + ascii_uppercase
digits = '0123456789'
hexdigits = digits + 'abcdef' + 'ABCDEF'
octdigits = '01234567'
punctuation = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
punctuation_nonPeriod = r"!\"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"
printable = digits + ascii_letters + punctuation + whitespace
unicode_str = '0123456789abcdefABCDEF\\xX'
LEMMATIZER = WordNetLemmatizer()


trans_table = {
    'whitespace': None,
    'ascii_lowercase' : ascii_uppercase,
    'ascii_uppercase' : ascii_uppercase,
    'ascii_letters' : ascii_letters,
    'digits':digits,
    # 'digits': {0:None,
    #            1:None,
    #            2:None,
    #            3:None,
    #            4:None,
    #            5:None,
    #            6:None,
    #            7:None,
    #            8:None,
    #            9:None},
    'hexdigits' : hexdigits,
    'octdigits' : octdigits,
    'punctuation' : punctuation,
    'printable' : printable
    }


class Preprocessor:


    def removeShortwords(self,text):
        text = re.sub(r'\b\w{,1}\b', '',text)

        return text

    def consecutiveDots(self,text):

        '''
        Removal of min 3 consecutive dots (periods)
        '''

        text = re.sub(r'\.{3,}',r' ', text)
        return text

    def decontracted_strings(self,phrase):
        '''
        Removal of Decontracted Strings

        '''
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def remove_esc_chars(text_string):
        '''
        Removes any escape character within text_string and returns the new string as type str.
        Keyword argument:
        - text_string: string instance
        Exceptions raised:
        - InputError: occurs should a non-string argument be passed
        '''

        if text_string is None or text_string == "":
            return ""
        elif isinstance(text_string, str):
            return " ".join(re.sub(r'\\\w', "", text_string).split())
        else:
            raise InputError("string not passed as argument")

    def removeUnicode(self,text):
        '''
        Removes unicode strings like "\u002c" and "x96"
        '''
        if isinstance(text,list):
            new_list = list()
            add = new_list.append
            for each in text:
                each = re.sub(r'(\\u[0-9A-Fa-f]+)',r' ', each)
                each = re.sub(r'[^\x00-\x7f]',r' ',each)
                each = re.sub(r'[^\x00-\x7f]',r' ',each)
                add(each)
            return new_list

        elif isinstance(text,str):

            text = re.sub(r'(\\u[0-9A-Fa-f]+)',r' ', text)
            text = re.sub(r'[^\x00-\x7f]',r' ',text)
            text = re.sub(r'[^\x00-\x7f]',r' ',text)

            return text

    def removehypen(self,text):

        '''
        Removal of Hypen inbetween words for a text corpus

        '''

        #text = self.consecutiveDots(text)
        empty_str = ''
        space_str = ' '
        #text = self.decontracted_strings(text)
        #translation = text.maketrans(space_str,space_str,'-')
        #translation = text.maketrans(empty_str,empty_str,punctuation_nonPeriod)
        # text = text.replace('[a-zA-Z]+\-[a-zA-Z]+',empty_str)
        # text = text.sub('[a-zA-Z]+\-[a-zA-Z]+',empty_str)
        #text = text.translate(translation)
        text = re.sub(r"([a-zA-Z])\-([a-zA-Z])", r"\1 \2", text)
        #translation = text.maketrans(trans_table['digits'],digits,digits)
        #text = text.translate(translation)
    #text = ''.join([i for i in text if not i.isdigit()])
        return text

    def removeNumbers(self,text,flag='digits'):
        '''
        Removal of Numbers words for a text corpus

        '''
        empty_str = ''

        translation = text.maketrans(empty_str,empty_str,digits)
        #translation = text.maketrans(trans_table['digits'],digits,digits)
        text = text.translate(translation)
    #text = ''.join([i for i in text if not i.isdigit()])
        return text

    def removePunctuation(self,text):

        '''
        Removal of Punctuations words for a text corpus
        Along with removal of Decontracted Strings [ Apostrafi's with a word ]

        '''

        #text = self.consecutiveDots(text)
        empty_str = ''
        text = self.decontracted_strings(text)
        translation = text.maketrans(empty_str,empty_str,punctuation)
        #translation = text.maketrans(trans_table['digits'],digits,digits)
        text = text.translate(translation)
    #text = ''.join([i for i in text if not i.isdigit()])
        return text

    def removeStopWords(self,inputText):

        '''
        Removal of stop words for a text corpus
        Stop words are cached from nltk module

        '''

        cachedStopWords = stopwords.words("english")
        #text = 'hello bye the the hi'
        text = ' '.join([word for word in inputText.split() if word not in cachedStopWords])
        return text

    def additional_text_removal(self,text):

        '''
        Additional Modifications and removals for the input text
        '''


        #text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"what's", "", text)
        text = re.sub(r"What's", "", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r" usa ", " America ", text)
        text = re.sub(r" USA ", " America ", text)
        text = re.sub(r" u s ", " America ", text)
        text = re.sub(r" uk ", " England ", text)
        text = re.sub(r" UK ", " England ", text)
        text = re.sub(r"india", "India", text)
        text = re.sub(r"switzerland", "Switzerland", text)
        text = re.sub(r"china", "China", text)
        text = re.sub(r"chinese", "Chinese", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r"quora", "Quora", text)
        text = re.sub(r" dms ", "direct messages ", text)
        text = re.sub(r"demonitization", "demonetization", text)
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r"KMs", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text)
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"gps", "GPS", text)
        text = re.sub(r"gst", "GST", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"dna", "DNA", text)
        text = re.sub(r"III", "3", text)
        text = re.sub(r"the US", "America", text)
        text = re.sub(r" J K ", " JK ", text)
        text = re.sub(r"Mr.", "Mr")
        text = re.sub(r"Mrs.", "Mrs")

        return text

    def _word_ngrams(self, tokens, stop_words=None):

        """
        Turn tokens into a sequence of n-grams after stop words filtering
        """

        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
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

            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
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


        return zip(input_list, input_list[1:])

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


    def preprocess_text(self,text_string, function_list,listFlag=False):
        '''
        Given each function within function_list, applies the order of functions put forward onto
        text_string, returning the processed string as type str.
        Keyword argument:
        - function_list: list of functions available in preprocessing.text
        - text_string: string instance

        Exceptions raised:

        - FunctionError: occurs should an invalid function be passed within the list of functions
        - InputError: occurs should text_string be non-string, or function_list be non-list
        - listFlag : Default True , returns text_string as a list
        '''
        # if isinstance(text_string,np.ndarray):
        #
        # elif text_string is None or text_string == "":
        #     return ""

        if isinstance(text_string, str):
            if isinstance(function_list, list):
                for func in function_list:
                    try:
                        text_string = func(text_string)
                    except (NameError, TypeError):
                        raise Exception("Invalid function passed - {0} as element of function_list.\nKindly check for valid functions".format(func))
                    except:
                        raise
                return text_string
            else:
                raise Exception("list of functions not passed as argument for function_list")
        elif isinstance(text_string, list):
            text_string = self.list2str(text_string)
            if isinstance(function_list, list):
                for func in function_list:
                    try:
                        text_string = func(text_string)
                    except (NameError, TypeError):
                        raise Exception("Invalid function passed as element of function_list.\nKindly check for valid functions")
                    except:
                        raise
                if listFlag:
                    return [x for x in text_string.split()]
                else:
                    return text_string
            else:
                raise Exception("list of functions not passed as argument for function_list")
        elif isinstance(text_string, np.ndarray):
            text_string = self.list2str(text_string)
            if isinstance(function_list, list):
                for func in function_list:
                    try:
                        text_string = func(text_string)
                    except (NameError, TypeError):
                        raise Exception("Invalid function passed as element of function_list.\nKindly check for valid functions")
                    except:
                        raise
                if listFlag:
                    return [','.join(x) for x in text_string.split()]
                else:
                    return text_string
            else:
                raise Exception("list of functions not passed as argument for function_list")
        else:
            raise Exception("string not passed as argument for text_string")

    def list2str(self,file_content):

        return ''.join([word.strip() for word in file_content])

    def lemmatize(text_string):
        '''
        Returns base from of text_string using NLTK's WordNetLemmatizer as type str.
        Keyword argument:
        - text_string: string instance
        Exceptions raised:
        - InputError: occurs should a non-string argument be passed
        '''

        if text_string is None or text_string == "":
            return ""
        elif isinstance(text_string, str):
            return LEMMATIZER.lemmatize(text_string)
        else:
            raise InputError("string not passed as primary argument")


    def lowercase(self,text_string):

        '''
        Converts string to lowercase
        '''

        if text_string is None or text_string == "":
            return ""
        elif isinstance(text_string, str):
            return text_string.lower()
        else:
            raise InputError("string not passed as argument for text_string")

    def remove_number_words(self,text_string):
        '''
        Removes any integer represented as a word within text_string and returns the new string as
        type str.
        Keyword argument:
        - text_string: string instance
        Exceptions raised:
        - InputError: occurs should a non-string argument be passed
        '''
        NUMBER_WORDS = ["zero","first","second","third","fourth","fifth","sixth","seventh","eigth","ninth","tenth","quarter","half","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","twenty-one","twenty-two","twenty-three","twenty-four","twenty-five","twenty-six","twenty-seven","twenty-eight","twenty-nine","thirty","thirty-one","thirty-two","thirty-three","thirty-four","thirty-five","thirty-six","thirty-seven","thirty-eight","thirty-nine","forty","forty-one","forty-two","forty-three","forty-four","forty-five","forty-six","forty-seven","forty-eight","forty-nine","fifty","fifty-one","fifty-two","fifty-three","fifty-four","fifty-five","fifty-six","fifty-seven","fifty-eight","fifty-nine","sixty","sixty-one","sixty-two","sixty-three","sixty-four","sixty-five","sixty-six","sixty-seven","sixty-eight","sixty-nine","seventy","seventy-one","seventy-two","seventy-three","seventy-four","seventy-five","seventy-six","seventy-seven","seventy-eight","seventy-nine","eighty","eighty-one","eighty-two","eighty-three","eighty-four","eighty-five","eighty-six","eighty-seven","eighty-eight","eighty-nine","ninety","ninety-one","ninety-two","ninety-three","ninety-four","ninety-five","ninety-six","ninety-seven","ninety-eight","ninety-nine","hundred","thousand","million","trillion","billion"]

        if text_string is None or text_string == "":
            return ""
        elif isinstance(text_string, str):
            for word in NUMBER_WORDS:
                text_string = re.sub(r'[\S]*\b'+word+r'[\S]*', "", text_string)
            return " ".join(text_string.split())
        else:
            raise InputError("String value not passed as argument")

#test_str = "Test string with 1456 Hello81286 hytt&89"

test_str1 = """One for all and all for one, Muskehounds are always ready. One for all and all for one, helping everybody. One for all and all for one, it's a pretty story. Sharing everything with fun, that's the way to be. One for all and all for one, Muskehounds are always ready. One for all and all for one, helping everybody. One for all and all for one, can sound pretty corny. If you've got a problem chum, think how it could be.

There's a voice that keeps on calling me. Down the road, that's where I'll always be. Every stop I make, I make a new friend. Can't stay for long, just turn around and I'm gone again. Maybe tomorrow, I'll want to settle down, Until tomorrow, I'll just keep moving on.

Hong Kong Phooey, number one super guy. Hong Kong Phooey, quicker than the human eye. He's got style, a groovy style, and a car that just won't stop. When the going gets tough, he's really rough, with a Hong Kong Phooey chop (Hi-Ya!). Hong Kong Phooey, number one super guy. Hong Kong Phooey, quicker than the human eye. Hong Kong Phooey, he's fan-riffic!

Thunder, thunder, thundercats, Ho! Thundercats are on the move, Thundercats are loose. Feel the magic, hear the roar, Thundercats are loose. Thunder, thunder, thunder, Thundercats! Thunder, thunder, thunder, Thundercats! Thunder, thunder, thunder, Thundercats! Thunder, thunder, thunder, Thundercats! Thundercats!
"""

test_str2 = """
As an advocate of Grammar I hereby produce the evidences to convince the importance of Punctuation , Honorable juries may decide on the merit of the case

Compare these sentences:

English Grammar teacher said, “Smith is useless.”

“English Grammar teacher” said Smith , “ is useless.”

Woman, without her man, is savage. (Man is given prominence.)

Woman ! without her, man is savage. (Woman is given importance.)

I’m sorry. You can’t come with us. (expresses lesser regret)

I’m sorry you can’t come with us.( expresses greater regret)

I did it . ( indicates a simple admission or confession)

I did it ? ( indicates a denial with strong doubt of the other person’s sanity implied)

I did it ! ( indicates the greatest achievement since Edison’s invention of the electric light )

My wife, who has had fever, is quite again.

Relative clauses define its antecedent……means that the writer has only one wife.

My wife who has had fever is quite again.

This Relative clauses dos not define its antecedent……means that the writer has more than one wife.

Oh! Never mind. ( Expresses strong emotion )

Oh, never mind. ( Expresses mild emotion. )

I’m sorry. You can’t come with us. (expresses lesser regret)

I’m sorry you can’t come with us.( expresses greater regret)

That’s all. I’ve finished. (indicates incomplete job)

That’s all I’ve finished.( indicates completed job)

I left him convinced he was fool. ( Means that I was convinced he was a fool )

I left him ,convinced he was fool. ( Means that I was able to convince him that he was a fool.)

Now, my friends, listen to me. ( means that the speaker was not able to change his friend’s mind)

Now my friends listen to me. ( means that the speaker has managed to change his friend’s mind)

I saw a man , eating lobster. ( saw a man )

I saw a man-eating lobster. ( saw a lobster)

Kill him , not leave him. ( Kill him )

Kill him not , leave him. ( do not kill him )

Let’s eat Grandma. (Trying to eat Grandma herself )

Let’s eat, Grandma. ( Trying to eat something not Grandma)
"""

pre_proc = Preprocessor()

#test = pre_proc.removeUnicode(test_str2)
#test = pre_proc.consecutiveDots(test)
#test_str2 = test_str2.encode('utf-8')
#test = pre_proc.consecutiveDots(test_str2)

#print(test_str2.encode('ascii','ignore'))#.decode('ascii'))
#quit()
#new_str = pre_proc.removeNumbers(test_str)

new_str1 = pre_proc.removeUnicode(test_str1)
new_str1 = pre_proc.removePunctuation(new_str1)
new_str1 = pre_proc.removeNumbers(new_str1)
new_str1 = pre_proc.removeStopWords(new_str1)
new_str1 = nltk.word_tokenize(new_str1)

new_str2 = pre_proc.removeUnicode(test_str2)
new_str2 = pre_proc.removeNumbers(new_str2)
new_str2 = pre_proc.removePunctuation(new_str2)
new_str2 = pre_proc.removeStopWords(new_str2)
new_str2 = nltk.word_tokenize(new_str2)

# print("Test Str = ", test_str,'\n\n')
# print("New Str = ", new_str)

input_list1 = ['all', 'this', 'happened', 'more', 'or', 'less']
input_list2 = ['all', 'this', 'happened', 'more', 'or', 'Databases','value','vaibhav']
input_list3 = ['more', 'or', 'Databases','value','vaibhav']

out2 = pre_proc.gen_ngrams_scikit_learn(input_list1,3,3)
#out = list(pre_proc.gen_bigrams(input_list))
out = list(pre_proc.generate_ngrams(input_list1,3))
#print(out)
#print(out2)
out3 = pre_proc.count_freq(input_list1)


input_ = [input_list1,input_list2,input_list3]
#input_ = [new_str1,new_str2]

#print(any(isinstance(el, list) for el in input_))

#quit()

input_list1 = str()
input_list2 = list()
input_file1 = "C:\\Users\\vsharm91\\Downloads\\txt_sentoken\\pos\\cv000_29590.txt"

with open(input_file1) as f:
    input_list1 = f.read().replace('\n', ' ')#splitlines()

input_file2 = "C:\\Users\\vsharm91\\Downloads\\txt_sentoken\\pos\\cv003_11664.txt"

with open(input_file2) as f:
    input_list2 = f.read().replace('\n', ' ')#.splitlines()



#input_list = [input_list1,input_list2]
#CountVector(input_)
#for each in input_list:
#text = pre_proc.preprocess_text(input_list2,[pre_proc.removeNumbers,pre_proc.removeStopWords,pre_proc.removePunctuation,pre_proc.lowercase])
#print(text)

#TFidfVector(input_)
#text = nltk.word_tokenize(text)
#df = CountVector(text)

#pprint(df)
