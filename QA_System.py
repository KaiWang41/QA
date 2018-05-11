import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


# def process_word(word):
#     word = word.lower()
#     lmtz = WordNetLemmatizer()
#     lemma = lmtz.lemmatize(word, 'v')
#     if lemma == word:
#         lemma = lmtz.lemmatize(word, 'n')
#     return lemma


OPEN_QUESTION_WORDS = ['what','who','whose','whom','where','when','why','how',
                       'which',"what's","who's","where's","how's"]
CLOSED_QUESTION_WORDS = ['is','are','am','was','were','do','does,','did','can',
                         'could','will','would','shall','should','have','has',
                         'had']


def to_lower(tokens):
    return [token.lower() for token in tokens]


def get_query_type(query):
    tokens = nltk.word_tokenize(query)
    tokens = to_lower(tokens)

    qword = ''
    for token in tokens:
        if token in OPEN_QUESTION_WORDS:
            qword = token
            if "'" in qword:
                qword = qword.split("'")[0]
            next_token = tokens[tokens.index(token)+1]
            break

    if qword == '':
        for (index, token) in enumerate(tokens):
            if token in CLOSED_QUESTION_WORDS:
                return 'closed'

    if qword == '':
        return 'undetermined'

    qtype = 'undetermined'
    if qword == 'when':
        qtype = 'time'
    elif qword == 'where':
        qtype = 'location'
    elif qword in ['who','whose','whom']:
        qtype = 'NE'
    # elif qword == 'how':
    #     if next


    return qtype


from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags

sentence = ""

ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
iob_tagged = tree2conlltags(ne_tree)
print(iob_tagged)
