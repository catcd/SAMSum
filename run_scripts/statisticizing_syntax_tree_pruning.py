import json
import logging
import os
import string

from nltk.parse.corenlp import CoreNLPParser

from utils.data_loader import DataLoader
from utils.pre_processing import nlp

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)

MIN_TOKENS = 2
MAX_TOKENS = 12
LENGTH_LIMIT = 500
MAIN_SIMILARITY_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.8

inside_count, outside_count = {}, {}


def is_valid_term(label, term, abstractive_summary):
    term_doc = nlp(term)
    for sentence in abstractive_summary.sents:
        if label in ['NP', 'VP']:
            if term_doc.similarity(sentence) >= MAIN_SIMILARITY_THRESHOLD:
                return True
        elif term_doc.similarity(sentence) >= SIMILARITY_THRESHOLD:
            return True
    return False


def dfs(node, abstractive_summary):
    if type(node[0]) is str:
        term = ''
        nodes_count = 0
        valid_terms = []
        for child in node:
            nodes_count += 1
            if len(term) == 0 or child._label in string.punctuation:
                term += child
            else:
                term += ' ' + child
        if MIN_TOKENS <= nodes_count <= MAX_TOKENS:
            outside_count[node._label] += 1
            if is_valid_term(node._label, term, abstractive_summary):
                valid_terms = [(node._label, term)]
        return term, nodes_count, valid_terms
    term = ''
    nodes_count = 0
    valid_terms = []
    for child in node:
        child_term, child_nodes_count, child_valid_terms = dfs(child, abstractive_summary)
        nodes_count += child_nodes_count
        valid_terms.extend(child_valid_terms)
        if len(term) == 0 or child._label in string.punctuation:
            term += child_term
        else:
            term += ' ' + child_term
    if MIN_TOKENS <= nodes_count <= MAX_TOKENS:
        if node._label not in outside_count:
            outside_count[node._label] = 0
        outside_count[node._label] += 1
        if is_valid_term(node._label, term, abstractive_summary):
            valid_terms = [(node._label, term)]
    return term, nodes_count, valid_terms


if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    parser = CoreNLPParser()
    for question_id, data in data_loader.docs.items():
        logging.info('Statisticizing for question {}'.format(question_id))
        extractive_summary = nlp(data['extractive_summary'])
        abstractive_summary = nlp(data['abstractive_summary'])
        for sentence in extractive_summary.sents:
            if len(sentence.text) > LENGTH_LIMIT:
                continue
            root = next(parser.raw_parse(sentence.text))
            _, _, valid_terms = dfs(root, abstractive_summary)
            for label, term in valid_terms:
                if label not in inside_count:
                    inside_count[label] = 0
                inside_count[label] += 1
    with open('./data/Statistics/Test_Labels.txt', 'w') as file:
        data = {
            'inside': inside_count,
            'outside': outside_count
        }
        data = json.dumps(data, indent=2)
        file.write(data)
