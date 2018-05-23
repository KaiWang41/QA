import spacy
from string import punctuation
#
# nlp = spacy.load('en_core_web_sm')
# doc = nlp('FTP stands for Foreign Tax Profit')
# for chunk in doc.noun_chunks:
#     print(chunk.text)

d = {1:1, 2:2, 3:9, 4:8, 5:6}

print(max(d, key=d.get))