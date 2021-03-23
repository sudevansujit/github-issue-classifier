
 
import spacy
nlp = spacy.load('training/model-last')


import pickle

pickle.dump(nlp, open('nlp.pkl', 'wb'))













