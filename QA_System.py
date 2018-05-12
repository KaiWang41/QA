import nltk
import json
import spacy
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_distances

OPEN_QUESTION_WORDS = ['what','who','whose','whom','where','when','why','how',
                       'which',"what's","who's","where's","how's"]
CLOSED_QUESTION_WORDS = ['is','are','am','was','were','do','does,','did','can',
                         'could','will','would','shall','should','have','has',
                         'had']
QUESTION_TYPES = ['PERSON','GPE','DATE','MONEY','CARDINAL','ORG','TIME',
                  'PERCENT','QUANTITY','ORDINAL']

stop = set(stopwords.words('english'))

with open('training.json') as json_data:
    train = json.load(json_data)

with open('documents.json') as json_data:
    documents = json.load(json_data)

nlp = spacy.load('en_core_web_sm')


def get_BOW_lower_nostop_alpha(sent):
    tokens = nltk.word_tokenize(sent)
    BOW = {}
    for token in tokens:
        token = token.lower()
        if token not in stop and token.isalpha():
            BOW[token] = BOW.get(token, 0) + 1
    return BOW


def get_qword(question):
    tokens = nltk.word_tokenize(question)
    for token in tokens:
        if token in OPEN_QUESTION_WORDS:
            return token
    for token in tokens:
        if token in CLOSED_QUESTION_WORDS:
            return token
    return 'others'


for train_case in train:
    question = train_case['question']
    docid = train_case['docid']
    para_num = train_case['answer_paragraph']

    para = documents[docid]['text'][para_num]

    sents = nltk.sent_tokenize(para)

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

    qword = get_qword(question)
    qtype = 'misc'
    if qword in ['who',"who's",'whose','whom']:
        qtype = 'PERSON'
    elif qword == 'when':
        qtype = 'DATE'
    elif qword in ['where',"where's"]:
        qtype = 'GPE'
    elif qword in ['how',"how's"]:
        tokens = nltk.word_tokenize(question)
        next_token = tokens[tokens.index('qword') + 1]
        if next_token == 'much':
            qtype = 'MONEY'
        elif next_token == 'many':
            qtype = 'CARDINAL'
        elif next_token == 'long':
            qtype = 'DATE'
        elif next_token == 'far':
            qtype = 'QUANTITY'
        elif next_token == 'old':
            qtype = 'DATE'
        else:
            doc = nlp(question)

    # elif qword in ['what']:


    break

doc = nlp('When was he born?')
for token in doc:
    print(token.subtree)