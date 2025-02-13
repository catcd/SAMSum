import logging
import os

from utils.data_loader import DataLoader
from utils.evaluation import evaluate_bert_score

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    predict, answer = [], []
    cnt = 0
    for question_id, data in data_loader.docs.items():
        predict.append(data['extractive_summary'])
        answer.append(data['abstractive_summary'])
    (P, R, F), hash_name = evaluate_bert_score(predict, answer)
    logging.info("F1: {}".format(F))
    logging.info(f"{hash_name}: P={P.mean().item():.12f} R={R.mean().item():.12f} F={F.mean().item():.12f}")
