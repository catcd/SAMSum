import logging
import os

from utils.data_loader import DataLoader

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.load_questions('./data/Test/TestSet_ShortQuestions.txt')
    data_loader.load_extractive_summaries('./data/Test/TestSet_MultiExtractiveSummaries.txt')
    data_loader.load_abstractive_summaries('./data/Test/TestSet_MultiAbstractiveSummaries.txt')
    data_loader.load_answers('./data/Test/TestSet_Answers.csv')
    data_loader.export_data('./data/Preprocessing/Test_Preprocessing.txt')
