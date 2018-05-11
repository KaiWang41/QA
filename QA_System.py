import nltk
import json
import spacy
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_distances

# OPEN_QUESTION_WORDS = ['what','who','whose','whom','where','when','why','how',
#                        'which',"what's","who's","where's","how's"]
# CLOSED_QUESTION_WORDS = ['is','are','am','was','were','do','does,','did','can',
#                          'could','will','would','shall','should','have','has',
#                          'had']

stop = set(stopwords.words('english'))

with open('training.json') as json_data:
    train = json.load(json_data)

with open('documents.json') as json_data:
    documents = json.load(json_data)

nlp = spacy.load('en')


def get_BOW_lower_nostop_alpha(sent):
    tokens = nltk.word_tokenize(sent)
    BOW = {}
    for token in tokens:
        token = token.lower()
        if token not in stop and token.isalpha():
            BOW[token] = BOW.get(token, 0) + 1
    return BOW


for train_case in train[0:2]:
    question = train_case['question']
    docid = train_case['docid']
    para_num = train_case['answer_paragraph']

    para = documents[docid]['text'][para_num]

    sents = nltk.sent_tokenize(para)
    # sents = []
    # for temp_sent in temp_sents:
    #     strings = temp_sent.split(',')
    #     sents += [s.strip() for s in strings]

    vectorizer = DictVectorizer()
    BOWs = []
    for sent in sents:
        BOW = get_BOW_lower_nostop_alpha(sent)
        BOWs.append(BOW)
    vector_space = vectorizer.fit_transform(BOWs)

    vector1 = vectorizer.transform(get_BOW_lower_nostop_alpha(question))

    sims = []

    for sent in sents:
        vector2 = vectorizer.transform(get_BOW_lower_nostop_alpha(sent))
        sim = 1 - cosine_distances(vector1, vector2)

        sims.append(sim)
        