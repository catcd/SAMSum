import logging
import os

from extractive_summarization.summarizer import Summarizer
from utils.data_loader import DataLoader

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing_en_core_web_lg.txt')
    summarizer = Summarizer(data_loader, params={
        'TFIDF_WEIGHT': 1,
        'LEXRANK_WEIGHT': 1,
        'KEYWORDS_WEIGHT': 1,
        'QUERY_BASED_WEIGHT': 5,
    })
    summarizer.scoring()
    summarizer.neighbors_boosting()
    summarizer.single_doc_summarizing(cutoff=15)
    summarizer.preprocessing_multi_docs()
    summarizer.multi_docs_summarizing(cutoff=30)
    summarizer.evaluate()
    summarizer.export_data('./data/Test/TestSet_MAS30_en_core_web_lg.txt')
