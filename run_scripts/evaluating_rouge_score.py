import logging
import os

from utils.data_loader import DataLoader
from utils.evaluation import evaluate_rouge

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)


def evaluate_from_single_file():
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    scores = {
        'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}
    }
    for question_id, data in data_loader.docs.items():
        extractive_summary = data['extractive_summary']
        abstractive_summary = data['abstractive_summary']
        score = evaluate_rouge(extractive_summary, abstractive_summary)
        for key, value in score.items():
            scores[key]['precision'] += value.precision
            scores[key]['recall'] += value.recall
            scores[key]['f1'] += value.fmeasure
    for k, v in scores.items():
        for kk, vv in v.items():
            scores[k][kk] /= len(data_loader.docs)
    logging.info(scores)


def evaluate_from_multi_files():
    data_loader = DataLoader()
    data_loader.load_extractive_summaries('./data/Test/TestSet_MultiExtractiveSummaries.txt')
    data_loader.load_abstractive_summaries('./data/Test/TestSet_MultiAbstractiveSummaries.txt')
    scores = {
        'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}
    }
    for question_id, question in data_loader.docs.items():
        abstractive_summary = question['abstractive_summary']
        with open('./data/Pegasus/{}.txt'.format(question_id), 'r') as file:
            pegasus_summary = file.read()
        score = evaluate_rouge(pegasus_summary, abstractive_summary)
        for key, value in score.items():
            scores[key]['precision'] += value.precision
            scores[key]['recall'] += value.recall
            scores[key]['f1'] += value.fmeasure
    for k, v in scores.items():
        for kk, vv in v.items():
            scores[k][kk] /= len(data_loader.docs)
    logging.info(scores)


if __name__ == '__main__':
    evaluate_from_single_file()
    # evaluate_from_multi_files()
