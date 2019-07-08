#!/usr/local/bin/python3

#########################
## author - Vaebhav
#########################

import sys
import os
sys.path.append(os.getcwd())
#from Feature_Extraction.extraction.text_extraction import unique_tokens
from extraction.text_extraction import unique_tokens
from extraction.text_extraction import BagOfWords
import pandas as pd

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


### UniGram Vector Space Generation.

UniGram_df = bowObj.createVectorSpace(token_sent,bagCount=0)

UniGram_df.to_excel('tests/BagOfWords_UniGramTest.xlsx')




### BiGram Vector Space Generation.
BiGram_df = bowObj.createVectorSpace(token_sent,bagCount=2)

BiGram_df.to_excel('tests/BagOfWords_BiGramTest.xlsx')
### Uncommment from here

#####
### Traditional BAG OF Words Vector Space using Scikit Learn Open Source Modules.
#####

### Both Modules produce Identical results with UniGram and BiGram approaches ( Bi Gram tokens are only considerd in BagOfWords).


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)

cv_matrix = cv.fit_transform(token_sent)

cv_matrix = cv_matrix.toarray()

vocab = cv.get_feature_names()

UniGram_CountVectorizer_df = pd.DataFrame(cv_matrix, columns=vocab)

UniGram_CountVectorizer_df.to_excel('tests/CountVectorizer_UniGramTest.xlsx')



cv = CountVectorizer(ngram_range=(1,2),min_df=0., max_df=1.)

cv_matrix = cv.fit_transform(token_sent)

cv_matrix = cv_matrix.toarray()

vocab = cv.get_feature_names()

BiGram_CountVectorizer_df = pd.DataFrame(cv_matrix, columns=vocab)

BiGram_CountVectorizer_df.to_excel('tests/CountVectorizer_BiGramTest.xlsx')
