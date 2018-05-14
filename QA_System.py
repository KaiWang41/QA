import nltk
import json
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_distances
from string import punctuation

OPEN_QUESTION_WORDS = ['what','who','whose','whom','where','when','why','how',
                       'which',"what's","who's","where's","how's"]
CLOSED_QUESTION_WORDS = ['is','are','am','was','were','do','does,','did','can',
                         'could','will','would','shall','should','have','has',
                         'had']
SIMILARITY_WEIGHT = 0.5
CORRECT_TYPE_WEIGHT = 0.5

stop = set(stopwords.words('english'))

with open('testing.json') as json_data:
    test = json.load(json_data)

with open('documents.json') as json_data:
    documents = json.load(json_data)

nlp = spacy.load('en_core_web_sm')


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


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
        if token.lower() in OPEN_QUESTION_WORDS:
            return token.lower()
    for token in tokens:
        if token.lower() in CLOSED_QUESTION_WORDS:
            return token.lower()
    return 'others'


case_count = 0
for test_case in test:
    question = test_case['question']
    docid = test_case['docid']

    doc = ''
    for para in documents[docid]['text']:
        doc += para + ' '

    sents = nltk.sent_tokenize(doc)

    vectorizer = DictVectorizer()
    BOWs = []
    for sent in sents:
        BOW = get_BOW_lower_nostop_alpha(sent)
        BOWs.append(BOW)
    vector_space = vectorizer.fit_transform(BOWs)

    vector1 = vectorizer.transform(get_BOW_lower_nostop_alpha(question))

    rankings = []
    for sent in sents:
        vector2 = vectorizer.transform(get_BOW_lower_nostop_alpha(sent))
        sim = 1 - cosine_distances(vector1, vector2)

        rankings.append(sim * SIMILARITY_WEIGHT)

    qword = get_qword(question)
    next_token = ''
    qtype = 'misc'
    dep = ''
    head = ''
    head_dep = ''
    subject = ''
    closed_q_choices = ('', '')
    doc = nlp(question)

    tokens = nltk.word_tokenize(question.lower())
    if qword in tokens:
        if tokens.index(qword) < len(tokens) - 1:
            next_token = tokens[tokens.index(qword) + 1]

    if qword in ['who',"who's",'whom']:
        qtype = 'who'
        for chunk in doc.noun_chunks:
            if qword in chunk.text:
                dep = chunk.root.dep_
                head = chunk.root.head.text
                head_dep = chunk.root.head.dep_

    elif qword == 'whose':
        qtype = 'PERSON'

    elif qword == 'when':
        qtype = 'when'

    elif qword in ['where',"where's"]:
        qtype = 'where'

    elif qword in ['how',"how's"]:
        if next_token == 'much':
            qtype = 'MONEY'
        elif next_token == 'many':
            qtype = 'CARDINAL'
        elif next_token == 'long':
            qtype = 'DATE'
        elif next_token in ['far','big','wide','deep','tall','high']:
            qtype = 'QUANTITY'
        elif next_token in ['old','young']:
            qtype = 'DATE'
        else:
            qtype = 'how'
            for token in doc:
                if token.text == qword:
                    head = token.head.text
                    head_dep = token.head.dep_
                    break

    elif qword in ['what', "what's", 'which']:
        if next_token in ['place','year','day','month','age','decade','century']:
            qtype = 'DATE'
        elif next_token == 'time':
            qtype = 'TIME'
        elif next_token in ['city','country','state']:
            qtype = 'GPE'
        elif next_token in ['place','river','mountain','ocean','sea','lake','continent','location']:
            qtype = 'LOC'
        elif next_token == 'percentage':
            qtype = 'PERCENT'
        elif next_token == 'value':
            qtype = 'QUANTITY'
        elif next_token == 'number':
            qtype = 'CARDINAL'
        elif next_token == 'price':
            qtype = 'MONEY'
        else:
            tag = nltk.pos_tag([next_token])[0][1]
            if tag in ['NN','NNS','NNP','NNPS']:
                qtype = 'noun'
                for chunk in doc.noun_chunks:
                    if qword in chunk.text:
                        dep = chunk.root.dep_
                        head = chunk.root.head.text
                        head_dep = chunk.root.head.dep_
                        break
            else:
                tokens.remove(next_token)
                if 'do' in tokens:
                    qtype = 'verb'
                    for token in doc:
                        if token.dep_ == 'nsubj':
                            subject = token.text
                            break
                else:
                    qtype = 'noun'
                    for chunk in doc.noun_chunks:
                        if qword in chunk.text:
                            dep = chunk.root.dep_
                            head = chunk.root.head.text
                            head_dep = chunk.root.head.dep_
                            break

    elif qword == 'why':
        qtype = 'why'

    elif qword in CLOSED_QUESTION_WORDS:
        qtype = 'closed'
        if 'or' in tokens:
            index = tokens.index('or')
            prev1 = tokens[index - 1]
            next1 = tokens[index + 1]
            tag_tokens = nltk.pos_tag(tokens)

            tag = tag_tokens[index - 1][1]
            if tag in ['NN', 'NNP', 'NNS', 'NNPS']:
                for chunk in doc.noun_chunks:
                    if prev1 in chunk.text:
                        first = chunk.text
                    if next1 in chunk.text:
                        second = chunk.text
                closed_q_choices = (first, second)
            else:
                closed_q_choices = (prev1, next1)
        else:
            qtype = 'others'

    answers = []
    max_ranking = -1
    max_idx = -1
    for (idx, sent) in enumerate(sents):
        doc = nlp(sent)
        has_possible_answer = 0
        answer = ''

        if qtype in ['GPE','DATE','TIME','PERCENT','QUANTITY','CARDINAL',
                     'MONEY','PERSON','ORG']:
            for ent in doc.ents:
                if ent.label_ == qtype and ent.text not in question:
                    has_possible_answer = 1
                    answer = ent.text

        elif qtype == 'when':
            for ent in doc.ents:
                if ent.label_ == 'TIME' or ent.label_ == 'DATE' and ent.text not in question:
                    has_possible_answer = 1
                    answer = ent.text

        elif qtype == 'where':
            for ent in doc.ents:
                if ent.label_ == 'GPE' or ent.label_ == 'LOC' and ent.text not in question:
                    has_possible_answer = 1
                    answer = ent.text

        elif qtype == 'how':
            for token in doc:
                if token.head.text == head and token.head.dep_ == head_dep:
                    has_possible_answer = 1
                    answer = token.text

        elif qtype == 'noun':
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ == dep and chunk.root.head.text == head and chunk.root.head.dep_ == head_dep and chunk.text not in question and chunk.text not in stop:
                    has_possible_answer = 1
                    answer = chunk.text

        elif qtype == 'who':
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and ent.text not in question:
                    has_possible_answer = 1
                    answer = ent.text

            if answer == '':
                for chunk in doc.noun_chunks:
                    if chunk.root.dep_ == dep and chunk.root.head.text == head and chunk.root.head.dep_ == head_dep and chunk.text not in question and chunk.text not in stop:
                        has_possible_answer = 1
                        answer = chunk.text

        elif qtype == 'verb':
            for token in doc:
                if token.dep_ == 'nsubj' and token.text == subject:
                    has_possible_answer = 1
                    for token1 in doc:
                        if token1.dep_ == 'ROOT' and token.text not in question:
                            answer = token1.text

        elif qtype == 'closed':
            first = closed_q_choices[0]
            second = closed_q_choices[1]
            appear1 = False
            appear2 = False
            negate1 = False
            negate2 = False
            neg_count = 0
            tokens = nltk.word_tokenize(sent)

            for (index, token) in enumerate(tokens):
                if token == 'not' or "n't" in token:
                    neg_count += 1

                    if index+1 < len(tokens):
                        if tokens[index+1] == first:
                            negate1 = True
                        if tokens[index+1] == second:
                            negate2 = True

                if token == first:
                    appear1 = True
                if token == second:
                    appear2 = True

            if (not appear1) and (not appear2):
                answer = second

            elif appear1 and not appear2:
                has_possible_answer = 1
                if neg_count % 2 == 1:
                    answer = second
                else:
                    answer = first

            elif appear2 and not appear1:
                has_possible_answer = 1
                if neg_count % 2 == 0:
                    answer = second
                else:
                    answer = first

            else:
                has_possible_answer = 1
                if negate1 and not negate2:
                    answer = second
                elif negate2 and not negate1:
                    answer = first
                else:
                    answer = second

        elif qtype == 'why':
            if 'reason' in sent or 'because' in sent or 'due to' in sent or 'since' in sent or 'for' in sent:
                has_possible_answer = 1

                if 'because of' in sent:
                    index = sent.index('because of')
                    substr = sent[index+11:]
                    span = nlp(substr)
                    for chunk in span.noun_chunks:
                        answer = chunk.text
                        break

                elif 'because' in sent:
                    index = sent.index('because')
                    substr = sent[index + 8:]
                    answer = strip_punctuation(substr[:-1])

                elif 'due to' in sent:
                    index = sent.index('due to')
                    substr = sent[index+7:]
                    span = nlp(substr)
                    for chunk in span.noun_chunks:
                        answer = chunk.text
                        break
                    if answer == '':
                        answer = strip_punctuation(substr[:-1])

                elif 'reason' in sent:
                    index = sent.index('reason')
                    substr = sent[index+7:]
                    span = nlp(substr)
                    for chunk in span.noun_chunks:
                        answer = chunk.text
                        break
                    if answer == '':
                        index = substr.find('is')
                        if index != -1:
                            answer = strip_punctuation(substr[index+3:-1])
                        else:
                            index = substr.find('are')
                            if index != -1:
                                answer = strip_punctuation(substr[index+4:-1])
                            else:
                                answer = strip_punctuation(sent[sent.index('reason'):-1])

                elif 'for' in sent:
                    index = sent.index('for')
                    substr = sent[index + 4:]
                    span = nlp(substr)
                    for chunk in span.noun_chunks:
                        answer = chunk.text
                        break
                    if answer == '':
                        answer = strip_punctuation(substr[:-1])

                elif 'since' in sent:
                    index = sent.index('since')
                    substr = sent[index + 6:]
                    answer = strip_punctuation(substr[:-1])

        rankings[idx] += has_possible_answer * CORRECT_TYPE_WEIGHT
        answers.append(answer)

        if rankings[idx] > max_ranking:
            max_ranking = rankings[idx]
            max_idx = idx

    answer = ''
    if answers[max_idx] != '':
        answer = answers[max_idx]
    else:
        sent = sents[max_idx]
        doc = nlp(sent)
        for chunk in doc.noun_chunks:
            if chunk.text not in question and chunk.text not in stop:
                answer = chunk.text
                break
        if answer == '':
            for chunk in doc.noun_chunks:
                if chunk.text not in question:
                    answer = chunk.text
                    break
        if answer == '':
            for chunk in doc.noun_chunks:
                answer = chunk.text
                break

    print(case_count, ',', answer.lower(), sep='')
    case_count += 1