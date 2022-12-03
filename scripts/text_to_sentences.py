import spacy

nlp = spacy.load('en_core_web_sm')
text = "You will find a rectangular white box in-front of you. open it. and take one packet from it. and put the packet in a cup. and then pour some boiled water in the cup"

doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
print(sentences)