import logging
import os
import re

from utils.data_loader import DataLoader
from utils.evaluation import evaluate_rouge
from utils.pre_processing import TextCleaner, TextSegmentator

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)


def remove_citations(text):
    return re.sub("[\[].*?[\]]", "", text)


def remove_brackets(text):
    text = text.replace('(', '')
    text = text.replace(')', '')
    return text


def remove_sentences_contain_urls(text):
    text_segmentator = TextSegmentator(text)
    sentences = text_segmentator.tokenize()
    results = []
    for key, sentence in sentences.items():
        if not bool(re.search(r'http\S+|www.\S+', sentence['sentence'])):
            results.append(sentence['sentence'])
    return ' '.join(results)


def remove_sentences_contain_urls(text):
    text_segmentator = TextSegmentator(text)
    sentences = text_segmentator.tokenize()
    results = []
    for key, sentence in sentences.items():
        if not bool(re.search(r'http\S+', sentence['sentence'])):
            results.append(sentence['sentence'])
    return ' '.join(results)


def remove_sentences_contain_phone_numbers(text):
    text_segmentator = TextSegmentator(text)
    sentences = text_segmentator.tokenize()
    results = []
    for key, sentence in sentences.items():
        if not bool(re.search(r'((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))', sentence['sentence'])):
            results.append(sentence['sentence'])
    return ' '.join(results)


def remove_repeat_sentences(text):
    THRESHOLD = 0.97
    text_segmentator = TextSegmentator(text)
    sentences = text_segmentator.tokenize()
    results = []
    for key, sentence in sentences.items():
        current_segmentator = TextSegmentator(sentence['sentence'])
        is_valid = True
        for previous_sentence in results:
            previous_segmentator = TextSegmentator(previous_sentence)
            if previous_segmentator.doc.similarity(current_segmentator.doc) >= THRESHOLD:
                is_valid = False
                break
        if is_valid:
            results.append(sentence['sentence'])
    return ' '.join(results)


if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    data_loader.load_extractive_summaries('./data/SyntaxTreePruning/Test_Pruning_Bart_GB.txt', delimiter='\t')
    scores = {
        'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}
    }
    summaries = []
    for question_id, question in data_loader.docs.items():
        logging.info('Post pruning for question {}'.format(question_id))
        predict_summary = question['extractive_summary']
        abstractive_summary = question['abstractive_summary']

        predict_summary = remove_citations(predict_summary)
        predict_summary = remove_brackets(predict_summary)
        predict_summary = remove_sentences_contain_urls(predict_summary)
        predict_summary = remove_sentences_contain_phone_numbers(predict_summary)
        predict_summary = remove_repeat_sentences(predict_summary)

        predict_summary = TextCleaner.remove_html_tags(predict_summary)
        predict_summary = TextCleaner.remove_whitespaces(predict_summary)
        score = evaluate_rouge(predict_summary, abstractive_summary)

        summaries.append(question_id + '\t' + predict_summary)

        for key, value in score.items():
            scores[key]['precision'] += value.precision
            scores[key]['recall'] += value.recall
            scores[key]['f1'] += value.fmeasure
    for k, v in scores.items():
        for kk, vv in v.items():
            scores[k][kk] /= len(data_loader.docs)
    logging.info(scores)
    with open('./data/SyntaxTreePruning/Test_Pruning_Bart_GB_Post.txt', 'w') as file:
        file.write('\n'.join(summaries))
