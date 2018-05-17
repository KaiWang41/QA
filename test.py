import spacy
from string import punctuation

nlp = spacy.load('en_core_web_sm')
doc = nlp('FTP stands for Foreign Tax Profit')
for chunk in doc.noun_chunks:
    print(chunk.text)
