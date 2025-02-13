# java -mx2g -cp ./stanford-corenlp-4.4.0/stanford-corenlp-4.4.0.jar:./stanford-corenlp-4.4.0/stanford-corenlp-4.4.0-models.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,parse,depparse -timeout 50000
import logging

from nltk.parse.corenlp import CoreNLPServer

logging.getLogger().setLevel(logging.INFO)

server = CoreNLPServer("./stanford-corenlp-4.4.0/stanford-corenlp-4.4.0.jar",
                       "./stanford-corenlp-4.4.0/stanford-corenlp-4.4.0-models.jar")
server.start()
