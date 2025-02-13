import json
import logging
import csv
import random

from utils.pre_processing import TextCleaner, TextSegmentator


class DataLoader:
    def __init__(self):
        self.docs = {}

    def load_questions(self, path='./data/Validation/ValidationSet_ShortQuestions.txt', delimiter='||'):
        logging.info('Load questions')
        with open(path, 'r') as file:
            questions = file.read().split('\n')
            for question in questions:
                question_id, question_text = question.split(delimiter)
                if question_id not in self.docs:
                    self.docs[question_id] = {}
                self.docs[question_id]['question'] = question_text
                # Preprocessing
                text_segmentator = TextSegmentator(question_text)
                self.docs[question_id]['question_ners'] = text_segmentator.get_ners()
                self.docs[question_id]['question_keywords'] = text_segmentator.get_keywords()

    def load_extractive_summaries(self, path='./data/Validation/ValidationSet_MultiExtractiveSummaries.txt', delimiter='||'):
        logging.info('Load extractive summaries')
        with open(path, 'r') as file:
            summaries = file.read().split('\n')
            for summary in summaries:
                question_id, extractive_summary = summary.split(delimiter)
                if question_id not in self.docs:
                    self.docs[question_id] = {}
                self.docs[question_id]['extractive_summary'] = extractive_summary

    def load_abstractive_summaries(self, path='./data/Validation/ValidationSet_MultiAbstractiveSummaries.txt', delimiter='||'):
        logging.info('Load abstractive summaries')
        with open(path, 'r') as file:
            summaries = file.read().split('\n')
            for summary in summaries:
                question_id, abstractive_summary = summary.split(delimiter)
                if question_id not in self.docs:
                    self.docs[question_id] = {}
                self.docs[question_id]['abstractive_summary'] = abstractive_summary

    def load_answers(self, path='./data/Validation/ValidationSet_Answers.csv'):
        with open(path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)
            for question_id, answer_id, answer in csv_reader:
                logging.info('Load answer {}'.format(answer_id))
                if question_id not in self.docs:
                    self.docs[question_id] = {}
                if 'answers' not in self.docs[question_id]:
                    self.docs[question_id]['answers'] = {}
                if answer_id not in self.docs[question_id]:
                    self.docs[question_id]['answers'][answer_id] = {}
                self.docs[question_id]['answers'][answer_id]['answer'] = answer
                # Preprocessing
                text = TextCleaner().clean(answer)
                text_segmentator = TextSegmentator(text)
                self.docs[question_id]['answers'][answer_id]['sentences'] = text_segmentator.tokenize()

    def load(self):
        self.load_questions()
        self.load_extractive_summaries()
        self.load_abstractive_summaries()
        self.load_answers()

    def export_data(self, path='./data/Preprocessing/Validation_Preprocessing.txt'):
        logging.info('Export preprocessing to {}'.format(path))
        with open(path, 'w') as file:
            data = json.dumps(self.docs, indent=2)
            file.write(data)
        logging.info('Export preprocessing done')

    def import_data(self, path='./data/Preprocessing/Validation_Preprocessing.txt'):
        logging.info('Import preprocessing from {}'.format(path))
        with open(path, 'r') as file:
            data = file.read()
            self.docs = json.loads(data)
