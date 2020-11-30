# from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import subjectivity
# from nltk.sentiment import SentimentAnalyzer
# from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import stanza # this is corenlp?
import spacy
from spacy import displacy

# == USING SPACY
nlp = spacy.load("en_core_web_sm")

doc = nlp("This bread is tasty.")

for token in doc:
    print(f'Token: {token.text} |', end=" ")
    print(f'Dependency: {token.dep_} |', end=" ")
    print(f'Lemma: {token.lemma_} |', end=" ")
    print(f'POS: {token.pos_} |', end=" ")
    print(f'HEAD: {token.head.text} |', end=" ")
    # print(f'Shape: {token.shape_} |', end=" ")
    # print(f'Is Alpha: {token.is_alpha} |', end=" ")
    # print(f'Is Stop: {token.is_stop} |', end=" ")
    print()
# ===== USING STANZA
#stanza.download('en') # get english


# ===== USING VADER
# sid = SentimentIntensityAnalyzer()
# ex_sent = ["very bad awful evil shill",
#            "I love fridays"]
#
# ss = sid.polarity_scores(ex_sent[0])
# print(ex_sent[0])
# for k in sorted(ss):
#     print('{0}: {1}, '.format(k, ss[k]), end='')
