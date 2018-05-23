import nltk
import json
import spacy
from nltk.corpus import stopwords
from math import log
from collections import defaultdict, Counter
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer

OPEN_QUESTION_WORDS = ['what','who','whose','whom','where','when','why','how',
                       'which',"what's","who's","where's","how's"]
CLOSED_QUESTION_WORDS = ['is','are','am','was','were','do','does,','did','can',
                         'could','will','would','shall','should','have','has',
                         'had']

# Stop words
stop = set(stopwords.words('english'))

lmtz = WordNetLemmatizer()

with open('testing.json') as json_data:
    test = json.load(json_data)

with open('documents.json') as json_data:
    documents = json.load(json_data)

# Spacy toolkit
nlp = spacy.load('en_core_web_sm')

punc = set(punctuation)


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punc)


def lemmatize(token):
    lemma = lmtz.lemmatize(token, 'v')
    if lemma == token:
        lemma = lmtz.lemmatize(token, 'n')
    return lemma


def extract_term_freqs(doc):
    tfs = {}
    for token in nltk.word_tokenize(doc):
        lemma = lemmatize(token.lower())
        if lemma not in stop and lemma.isalpha():
            tfs[lemma] = tfs.get(lemma, 0) + 1
    return tfs


def compute_doc_freqs(doc_term_freqs):
    dfs = Counter()
    for tfs in doc_term_freqs.values():
        for term in tfs.keys():
            dfs[term] += 1
    return dfs


def query_vsm(query, index, k=20):
    accumulator = Counter()
    for term in query:
        postings = index[term]
        for docid, weight in postings:
            accumulator[docid] += weight
    return accumulator.most_common(k)


# Find the question word
def get_qword(question):
    tokens = nltk.word_tokenize(question.lower())
    for token in tokens:
        if token in OPEN_QUESTION_WORDS:
            return token
    for token in tokens:
        if token in CLOSED_QUESTION_WORDS:
            return token
    return 'others'


# length of longest same sequences of keywords
def get_overlap(sent1, sent2):
    tokens1 = []
    tokens2 = []

    for token in nltk.word_tokenize(strip_punctuation(sent1.lower())):
        lemma = lemmatize(token)
        if lemma not in stop:
            tokens1.append(lemma)

    for token in nltk.word_tokenize(strip_punctuation(sent2.lower())):
        lemma = lemmatize(token)
        if lemma not in stop:
            tokens2.append(lemma)

    max = 0
    for i in range(len(tokens1)):
        for j in range(len(tokens2)):

            if tokens1[i] == tokens2[j]:
                length = 1

                ii = i + 1
                jj = j + 1
                while ii < len(tokens1) and jj < len(tokens2) and \
                        tokens1[ii] == tokens2[jj]:
                    ii += 1
                    jj += 1
                    length += 1

                if length > max:
                    max = length

    return max


file = open('Team_Strong.csv', 'w')
file.write('id,answer\n')

case_count = 0
# test = [test[17]]
for test_case in test:
    question = test_case['question']
    docid = test_case['docid']

    # Convert doc into one string, then tokenize sentences
    corpus = ''
    for para in documents[docid]['text']:
        corpus += para + ' '

    # sentence as a document
    raw_docs = nltk.sent_tokenize(corpus)

    # TFIDF
    doc_term_freqs = {}
    for (id, raw_doc) in enumerate(raw_docs):
        term_freqs = extract_term_freqs(raw_doc)
        doc_term_freqs[id] = term_freqs
    M = len(doc_term_freqs)

    doc_freqs = compute_doc_freqs(doc_term_freqs)

    vsm_inverted_index = defaultdict(list)
    for docid, term_freqs in doc_term_freqs.items():
        N = sum(term_freqs.values())
        length = 0

        # find tf*idf values and accumulate sum of squares
        tfidf_values = []
        for term, count in term_freqs.items():
            tfidf = float(count) / N * log(M / float(doc_freqs[term]))
            tfidf_values.append((term, tfidf))
            length += tfidf ** 2

        # normalise documents by length and insert into index
        length = length ** 0.5
        for term, tfidf in tfidf_values:
            # inversion of the indexing, term -> (doc_id, score)
            vsm_inverted_index[term].append([docid, tfidf / length])

    for term, docids in vsm_inverted_index.items():
        docids.sort()

    terms = extract_term_freqs(question)
    results = query_vsm(terms, vsm_inverted_index)


    # Step 2
    # Analyse question type
    qword = get_qword(question)

    # the word after question word, such as 'what value', 'which gender'
    next_token = ''

    qtype = ''

    # dependency parsing
    dep = ''

    # head word
    head = ''

    # head dependency
    head_dep = ''

    # subject, root, object
    nsubj = ''
    ROOT = ''
    dobj = ''

    # yes or no questions have two options
    closed_q_choices = ('', '')

    doc = nlp(question)

    tokens = nltk.word_tokenize(question.lower())

    # get next word
    if qword in tokens:
        if tokens.index(qword) < len(tokens) - 1:
            next_token = tokens[tokens.index(qword) + 1]

    # get structure of sentence
    for token in doc:
        if 'nsubj' in token.dep_:
            nsubj = lemmatize(strip_punctuation(token.text))
        if token.dep_ == 'ROOT':
            ROOT = lemmatize(strip_punctuation(token.text))
        if 'dobj' in token.dep_:
            dobj = lemmatize(strip_punctuation(token.text))

    # for noun (phrase) questions, get answer dependency
    for chunk in doc.noun_chunks:
        if qword in chunk.text:
            dep = chunk.root.dep_
            head = lemmatize(strip_punctuation(chunk.root.head.text))
            head_dep = chunk.root.head.dep_

    # determine answer type
    if 'stand for' in question or 'abbreviat' in question:
        qtype = 'abrv'

    elif qword in ['who',"who's",'whom','whose']:
        qtype = 'who'

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
        elif next_token in ['far','big','wide','deep','tall','high','fast','heavy']:
            qtype = 'QUANTITY'
        elif next_token in ['old','young']:
            qtype = 'DATE'
        elif next_token in ['does','did','do','have','has','had','should',
                              'can','could','will','would','must']:
            if dobj != '':
                qtype = 'adj'
            else:
                qtype = 'verb'

    elif qword in ['what', "what's", 'which']:

        if 'year'in tokens or \
                'day' in tokens or \
                'month' in tokens or \
                'era' in tokens or \
                'age' in tokens or \
                'century' in tokens or \
                'week' in tokens or \
                'period' in tokens or \
                'dynasty' in tokens:
            qtype = 'DATE'

        elif 'company' in tokens or \
                'organization' in tokens or \
                'organisation' in tokens or \
                'corporation' in tokens or \
                'institution' in tokens or \
                'university' in tokens or \
                'corporation' in tokens or \
                'association' in tokens or \
                'union' in tokens or \
                'agency' in tokens:
            qtype = 'ORG'

        elif 'city' in tokens or \
                'country' in tokens or \
                'state' in tokens or \
                'province' in tokens or \
                'county' in tokens:
            qtype = 'GPE'

        elif 'place' in tokens or \
                'river' in tokens or \
                'mountain' in tokens or \
                'ocean' in tokens or \
                'region' in tokens or \
                'area' in tokens or \
                'sea' in tokens or \
                'lake' in tokens or \
                'continent' in tokens or \
                'location' in tokens or \
                'forest' in tokens or \
                'jungle' in tokens:
            qtype = 'LOC'

        elif 'nationality' in tokens:
            qtype = 'NORP'

        elif 'building' in tokens or \
            'airport' in tokens or \
            'highway' in tokens or \
            'bridge' in tokens or \
            'harbour' in tokens or \
            'harbor' in tokens or \
            'port' in tokens or \
            'dam' in tokens:
            qtype = 'FACILITY'

        elif 'hurricane' in tokens or \
            'battle' in tokens or \
            'war' in tokens:
            qtype = 'EVENT'

        elif 'book' in tokens or \
            'novel' in tokens or \
            'song' in tokens or \
            'music' in tokens or \
            'painting' in tokens:
            qtype = 'WORK_OF_ART'

        elif 'language' in tokens or \
                'speak' in tokens:
            qtype = 'LANGUAGE'

        elif 'percentage' in tokens or 'percent' in tokens:
            qtype = 'PERCENT'

        elif 'value' in tokens or \
                'distance' in tokens or \
                'size' in tokens or \
                'length' in tokens or \
                'depth' in tokens or \
                'height' in tokens or \
                'density' in tokens or \
                'speed' in tokens or \
                'weight' in tokens or \
                'area' in tokens or \
                'temperature' in tokens or \
                'volume' in tokens:
            qtype = 'QUANTITY'

        elif 'number' in tokens:
            qtype = 'CARDINAL'

        elif 'price' in tokens:
            qtype = 'MONEY'

        else:
            # what...do type question
            tokens.remove(next_token)
            if 'do' in tokens:
                qtype = 'verb'
            else:
                qtype = 'noun'

    elif qword == 'why':
        qtype = 'why'

    elif qword in CLOSED_QUESTION_WORDS:
        qtype = 'closed'

        # answer is one of the 'or' options in the question
        if 'or' in tokens:
            index = tokens.index('or')
            prev1 = tokens[index - 1]
            next1 = tokens[index + 1]
            tag_tokens = nltk.pos_tag(tokens)

            tag = tag_tokens[index - 1][1]

            # if answer is a noun
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

    # re-rank the 20 sentences
    scores = {}
    for id, _ in results:
        sent = raw_docs[id]
        doc = nlp(sent)

        score = get_overlap(sent, question)

        if qtype == 'who':
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    score += 1

        elif qtype == 'when':
            for ent in doc.ents:
                if ent.label_ == 'TIME' or ent.label_ == "DATE":
                    score += 1

        elif qtype == 'where':
            for ent in doc.ents:
                if ent.label_ == 'GPE' or ent.label_ == "LOC":
                    score += 1

        elif qtype in ['LANGUAGE','WORK_OF_ART','EVENT','NORP','FACILITY',
                       'GPE','DATE','TIME','PERCENT','QUANTITY','CARDINAL',
                     'MONEY','PERSON','ORG','LOC']:
            for ent in doc.ents:
                if ent.label_ == qtype:
                    score += 1

        elif qtype == 'adj':
            for token in doc:
                if 'advmod' in token.dep_ or 'acomp' in token.dep_:
                    score += 1

        elif qtype == 'verb':
            for token in doc:
                if token.dep_ == 'ROOT':
                    score += 1

        elif qtype == 'closed':
            first = closed_q_choices[0]
            second = closed_q_choices[1]

            score += (first in sent) + (second in sent)

        elif qtype == 'why':
            if 'reason' in sent or 'because' in sent or 'due to' in sent or 'since' in sent or 'for' in sent:
                score += 1

        scores[id] = score

    rank = {}
    for id, sim in results:
        max_score = scores[max(scores, key=scores.get)]
        if max_score != 0:
            rank[id] = sim * 0.5 + (scores[id] / max_score * 0.5)
        else:
            rank[id] = sim
    
    # sentence with highest rank
    index = max(rank, key=rank.get)
    sent = raw_docs[index]
    doc = nlp(sent)

    # find sentence structure
    sent_nsubj = ''
    sent_ROOT = ''
    sent_dobj = ''
    for token in doc:
        if 'nsubj' in token.dep_:
            sent_nsubj = lemmatize(strip_punctuation(token.text))
        if token.dep_ == 'ROOT':
            sent_ROOT = lemmatize(strip_punctuation(token.text))
        if 'dobj' in token.dep_:
            sent_dobj = lemmatize(strip_punctuation(token.text))
            
    # find answer with highest score
    max_score = -1
    answer = ''
    
    if qtype == 'who':
        for np in doc.noun_chunks:
            score = 0
            
            if np in doc.ents:
                for ent in doc.ents:
                    if np.text in ent.text and ent.label_ == 'PERSON':
                            score += 3

            # find NP dependency
            np_dep = np.root.dep_
            np_head = lemmatize(strip_punctuation(np.root.head.text))
            np_head_dep = np.root.head.dep_

            if np_dep == dep:
                score += 1
            if np_head == head:
                score += 1
            if np_head_dep == head_dep:
                score += 1

            if np.text not in question:
                score += 1

            if strip_punctuation(np.text).strip().lower() not in stop:
                score += 1

            if np.text.lower() == 'it':
                score = -1
                
            if score > max_score:
                max_score = score
                answer = np.text

    elif qtype == 'when':
        for ent in doc.ents:
            score = 0
            
            if ent.label_ == 'TIME' or ent.label_ == "DATE":
                score += 3
                
            if ent.text not in question:
                score += 1
            
            if score > max_score:
                max_score = score
                answer = ent.text

    elif qtype == 'where':
        for ent in doc.ents:
            score = 0
            
            if ent.label_ == 'GPE' or ent.label_ == "LOC":
                score += 3

            if ent.text not in question:
                score += 1
                
            if score > max_score:
                max_score = score
                answer = ent.text
            
    elif qtype in ['LANGUAGE', 'WORK_OF_ART', 'EVENT', 'NORP', 'FACILITY',
                   'GPE', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL',
                   'MONEY', 'PERSON', 'ORG', 'LOC']:
        for ent in doc.ents:
            score = 0
            
            if ent.label_ == qtype:
                score += 3
            
            if ent.text not in question:
                score += 1
                
            if score > max_score:
                max_score = score
                answer = ent.text

    elif qtype == 'abrv':
        abrv = ''
        qdoc = nlp(question)
        for token in qdoc:
            text = token.text
            if len(text) >= 2 and text.isupper() and text.isalpha():
                abrv = text.lower()

        if abrv == '' and 'stand for' in question:
            tokens = question.lower().split(' ')
            abrv = tokens[tokens.index('stand')-1]

        if abrv != '':
            tokens = nltk.word_tokenize(sent)
            for (i, token) in enumerate(tokens):
                if token[0].isupper():
                    k = 1
                    phrase = token.lower()
                    initials = phrase[0]

                    while i+k < len(tokens) and tokens[i+k][0].isupper():
                        phrase = phrase + ' ' + tokens[i+k].lower()
                        initials += tokens[i+k][0].lower()
                        k += 1

                    phrase = phrase.strip()
                    if initials == abrv:
                        answer = phrase

        else:
            tokens = nltk.word_tokenize(question)
            for (i, token) in enumerate(tokens):
                if token[0].isupper():
                    k = 1
                    initials = token[0].lower()

                    while i + k < len(tokens) and tokens[i + k][0].isupper():
                        initials += tokens[i + k][0].lower()
                        k += 1

                    if len(initials) >= 2:
                        answer = initials

    elif qtype == 'adj':
        for token in doc:
            score = 0
            
            if 'advmod' in token.dep_ or 'acomp' in token.dep_:
                score += 3

            token_dep = token.dep_
            token_head = lemmatize(strip_punctuation(token.head.text))
            token_head_dep = token.head.dep_

            if token_dep == dep:
                score += 1
            if token_head == head:
                score += 1
            if token_head_dep == head_dep:
                score += 1

            if token.text not in question:
                score += 1

            if strip_punctuation(token.text).strip().lower() not in stop:
                score += 1

            if token.text.lower() == 'it':
                score = -1

            if score > max_score:
                max_score = score
                answer = token.text

    elif qtype == 'verb':
        for token in doc:
            score = 0

            if token.dep_ == 'ROOT':
                score += 1

            if lemmatize(strip_punctuation(token.text)) not in \
                    [lemmatize(strip_punctuation(s)) for s in nltk.word_tokenize(question)]:
                score += 1

            if strip_punctuation(token.text).strip().lower() not in stop:
                score += 1

            if score > max_score:
                max_score = score
                answer = token.text

    elif qtype == 'closed':
        first = closed_q_choices[0]
        second = closed_q_choices[1]

        # whether each option appears (and is negates)
        appear1 = False
        appear2 = False
        negate1 = False
        negate2 = False
        neg_count = 0
        tokens = nltk.word_tokenize(raw_docs[id])

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

        possible_answer = ''
        if appear1 and not appear2:
            if neg_count % 2 == 1:
                possible_answer = second
            else:
                possible_answer = first

        elif appear2 and not appear1:
            if neg_count % 2 == 0:
                possible_answer = second
            else:
                possible_answer = first

        elif appear1 and appear2:
            if negate1 and not negate2:
                possible_answer = second
            elif negate2 and not negate1:
                possible_answer = first
            else:
                possible_answer = second

        if possible_answer != '':
            score += 5
            if score > max_score:
                max_score = score
                answer = possible_answer

    elif qtype == 'why':

        possible_answer = ''
        score = 0

        if 'reason' in sent or 'because' in sent or 'due to' in sent or 'since' in sent or 'for' in sent:

            if 'because of' in sent:
                score += 3
                index = sent.index('because of')
                substr = sent[index+11:]
                span = nlp(substr)
                for chunk in span.noun_chunks:
                    possible_answer = chunk.text
                    break

            elif 'because' in sent:
                score += 3
                index = sent.index('because')
                substr = sent[index + 8:]
                possible_answer = substr

            elif 'due to' in sent:
                score += 3
                index = sent.index('due to')
                substr = sent[index+7:]
                span = nlp(substr)
                for chunk in span.noun_chunks:
                    possible_answer = chunk.text
                    break
                if possible_answer == '':
                    possible_answer = substr

            elif 'reason' in sent:
                score += 2
                index = sent.index('reason')
                substr = sent[index+7:]
                span = nlp(substr)
                for chunk in span.noun_chunks:
                    possible_answer = chunk.text
                    break
                if possible_answer == '':
                    index = substr.find('is')
                    if index != -1:
                        possible_answer = substr[index+3]
                    else:
                        index = substr.find('was')
                        if index != -1:
                            possible_answer = substr[index+4]
                        else:
                            possible_answer = sent[sent.index('reason'):]

            elif 'for' in sent:
                score += 1
                index = sent.index('for')
                substr = sent[index + 4:]
                span = nlp(substr)
                for chunk in span.noun_chunks:
                    possible_answer = chunk.text
                    break
                if possible_answer == '':
                    possible_answer = substr

            elif 'since' in sent:
                score += 1
                index = sent.index('since')
                substr = sent[index + 6:]
                possible_answer = substr

            if possible_answer != '' and score > max_score:
                answer = possible_answer
                max_score = score

    # if answer not found, find noun phrases
    if answer == '':
        for np in doc.noun_chunks:
            score = 0

            np_dep = np.root.dep_
            np_head = lemmatize(strip_punctuation(np.root.head.text))
            np_head_dep = np.root.head.dep_

            if np_dep == dep:
                score += 1
            if np_head == head:
                score += 1
            if np_head_dep == head_dep:
                score += 1

            if np.text not in question:
                score += 1

            if strip_punctuation(np.text).strip().lower() not in stop:
                score += 1

            if np.text.lower() == 'it':
                score = 0

            if score > max_score:
                max_score = score
                answer = np.text

    file.write(str(case_count))
    file.write(',')
    file.write(strip_punctuation(answer).strip().lower())
    file.write('\n')
    print(case_count,' ',answer)
    case_count += 1

file.close()