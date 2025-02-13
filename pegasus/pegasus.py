import sys
import argparse
import logging
import os

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

sys.path.append("..")

from utils.data_loader import DataLoader
from utils.evaluation import evaluate_rouge

os.chdir('..')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('pegasus/logging_info.txt'),
        logging.StreamHandler()
    ]
)


class Pegasus:
    RANGE_RATIO = 0.4
    MIN_LENGTH_RATIO = 1
    MAX_LENGTH_RATIO = 1.5

    def __init__(self,
                 model_path='./pegasus/pegasus_large',
                 extractive_path='./data/Test/TestSet_MultiExtractiveSummaries.txt',
                 abstractive_path='./data/Test/TestSet_MultiAbstractiveSummaries.txt'):
        self.data_loader = DataLoader()
        self.data_loader.load_extractive_summaries(extractive_path)
        self.data_loader.load_abstractive_summaries(abstractive_path)
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info('Loading PEGASUS model')
        self.tokenizer = PegasusTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_path).to(torch_device)

    def summarize(self, question_id):
        logging.info('Abstractive summarize for question {}'.format(question_id))
        abstractive_summary = self.data_loader.docs[question_id]['abstractive_summary']
        extractive_summary = self.data_loader.docs[question_id]['extractive_summary']
        tokens = self.tokenizer(extractive_summary, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        extractive_tokens_count = len(extractive_summary.split(' '))
        abstractive_tokens_count = len(abstractive_summary.split(' '))
        min_length_ratio = min(abstractive_tokens_count / extractive_tokens_count + self.RANGE_RATIO, self.MIN_LENGTH_RATIO)
        logging.info('Min length ratio: {}'.format(min_length_ratio))
        min_length = int(extractive_tokens_count * min_length_ratio)
        max_length = int(extractive_tokens_count * self.MAX_LENGTH_RATIO)
        encoded_tokens = self.model.generate(**tokens,
                                             min_length=min_length,
                                             max_length=max_length,
                                             no_repeat_ngram_size=3,
                                             num_beams=5)[0]
        pegasus_summary = self.tokenizer.decode(encoded_tokens, skip_special_tokens=True)
        with open('./data/Pegasus/{}.txt'.format(question_id), 'w') as file:
            file.write(pegasus_summary)
        logging.info('Summary for question {}: {}'.format(question_id, pegasus_summary))
        extractive_score = evaluate_rouge(extractive_summary, abstractive_summary)
        pegasus_score = evaluate_rouge(pegasus_summary, abstractive_summary)
        logging.info('Extractive score: {}'.format(extractive_score))
        logging.info('PEGASUS score: {}'.format(pegasus_score))


if __name__ == '__main__':
    # Loading arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', '-q', type=str, help='List of Question IDs, separated by commas')
    args = parser.parse_args()
    question_ids = args.questions.split(',')
    # Summarizing
    pegasus = Pegasus()
    for question_id in question_ids:
        pegasus.summarize(question_id)
