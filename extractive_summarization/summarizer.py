import logging

from extractive_summarization.scores import TfidfScorer, LexrankScorer, KeywordsScorer, QueryBasedScorer
from utils.evaluation import evaluate_rouge
from utils.pre_processing import TextSegmentator


class Summarizer:
    TFIDF_WEIGHT = 1
    LEXRANK_WEIGHT = 1
    KEYWORDS_WEIGHT = 1
    QUERY_BASED_WEIGHT = 1

    def __init__(self, data_loader, params=None):
        self.data_loader = data_loader
        self.scores = {}
        self.single_docs = {}
        self.multi_docs = {}
        self.multi_docs_scores = {}
        self.answers = {}
        if params:
            if 'TFIDF_WEIGHT' in params:
                self.TFIDF_WEIGHT = params['TFIDF_WEIGHT']
            if 'LEXRANK_WEIGHT' in params:
                self.LEXRANK_WEIGHT = params['LEXRANK_WEIGHT']
            if 'KEYWORDS_WEIGHT' in params:
                self.KEYWORDS_WEIGHT = params['KEYWORDS_WEIGHT']
            if 'QUERY_BASED_WEIGHT' in params:
                self.QUERY_BASED_WEIGHT = params['QUERY_BASED_WEIGHT']

    def scoring(self):
        # Scoring
        for question_id, question in self.data_loader.docs.items():
            logging.info('Scoring for question {}'.format(question_id))
            self.scores[question_id] = {}
            self.scores[question_id]['tfidf'] = TfidfScorer.scoring(question)
            self.scores[question_id]['lexrank'] = LexrankScorer.scoring(question)
            self.scores[question_id]['keywords'] = KeywordsScorer.scoring(question)
            self.scores[question_id]['query_based'] = QueryBasedScorer.scoring(question)
        # Combining
        logging.info('Combining scores')
        for question_id, question in self.data_loader.docs.items():
            self.scores[question_id]['final'] = {}
            for answer_id, answer in question['answers'].items():
                self.scores[question_id]['final'][answer_id] = {}
                for sentence_id, sentence in answer['sentences'].items():
                    self.scores[question_id]['final'][answer_id][sentence_id] = \
                        self.TFIDF_WEIGHT * self.scores[question_id]['tfidf'][answer_id][sentence_id] \
                        + self.LEXRANK_WEIGHT * self.scores[question_id]['lexrank'][answer_id][sentence_id] \
                        + self.KEYWORDS_WEIGHT * self.scores[question_id]['keywords'][answer_id][sentence_id] \
                        + self.QUERY_BASED_WEIGHT * self.scores[question_id]['query_based'][answer_id][sentence_id]

    def neighbors_boosting(self, clusters=5, boost_range=3):
        logging.info('Neighbors boosting')
        for question_id, question in self.data_loader.docs.items():
            self.scores[question_id]['boost'] = self.scores[question_id]['final']
            for answer_id, answer in question['answers'].items():
                # Finding some best sentences
                pairs = []
                for sentence_id, sentence in answer['sentences'].items():
                    pairs.append((self.scores[question_id]['final'][answer_id][sentence_id], sentence_id))
                pairs = sorted(pairs, reverse=True)
                if len(pairs) > clusters:
                    pairs = pairs[:clusters]
                # Boosting their relatives
                for score, pivot in pairs:
                    pivot = int(pivot)
                    for sentence_id, sentence in answer['sentences'].items():
                        if abs(int(sentence_id) - pivot) <= boost_range:
                            self.scores[question_id]['boost'][answer_id][sentence_id] += score

    def single_doc_summarizing(self, cutoff=10):
        logging.info('Single docs summarizing')
        for question_id, question in self.data_loader.docs.items():
            self.single_docs[question_id] = {}
            for answer_id, answer in question['answers'].items():
                # Getting some best sentences
                pairs = []
                for sentence_id, sentence in answer['sentences'].items():
                    pairs.append((self.scores[question_id]['boost'][answer_id][sentence_id], sentence_id))
                pairs = sorted(pairs, reverse=True)
                if len(pairs) > cutoff:
                    pairs = pairs[:cutoff]
                # Reordering to original order
                pairs = sorted(pairs, key=lambda x: x[1])
                summary = ' '.join([answer['sentences'][sentence_id]['sentence'] for score, sentence_id in pairs])
                self.single_docs[question_id][answer_id] = summary

    def preprocessing_multi_docs(self):
        for question_id, question in self.single_docs.items():
            logging.info('Multi docs preprocessing for question {}'.format(question_id))
            summaries = ' '.join([answer for answer_id, answer in question.items()])
            self.multi_docs[question_id] = {}
            self.multi_docs[question_id]['question'] = self.data_loader.docs[question_id]['question']
            self.multi_docs[question_id]['question_ners'] = self.data_loader.docs[question_id]['question_ners']
            self.multi_docs[question_id]['question_keywords'] = self.data_loader.docs[question_id]['question_keywords']
            text_segmentator = TextSegmentator(summaries)
            self.multi_docs[question_id]['answers'] = {}
            self.multi_docs[question_id]['answers']['{}_Answer1'.format(question_id)] = {}
            self.multi_docs[question_id]['answers']['{}_Answer1'.format(question_id)]['sentences'] = text_segmentator.tokenize()

    def multi_docs_summarizing(self, cutoff=20):
        # Scoring
        for question_id, question in self.multi_docs.items():
            logging.info('Multi docs scoring for question {}'.format(question_id))
            self.multi_docs_scores[question_id] = {}
            self.multi_docs_scores[question_id]['tfidf'] = TfidfScorer.scoring(question)
            self.multi_docs_scores[question_id]['lexrank'] = LexrankScorer.scoring(question)
            self.multi_docs_scores[question_id]['keywords'] = KeywordsScorer.scoring(question)
            self.multi_docs_scores[question_id]['query_based'] = QueryBasedScorer.scoring(question)
        # Combining
        logging.info('Combining scores')
        for question_id, question in self.multi_docs.items():
            self.multi_docs_scores[question_id]['final'] = {}
            for answer_id, answer in question['answers'].items():
                self.multi_docs_scores[question_id]['final'][answer_id] = {}
                for sentence_id, sentence in answer['sentences'].items():
                    self.multi_docs_scores[question_id]['final'][answer_id][sentence_id] = \
                        self.TFIDF_WEIGHT * self.multi_docs_scores[question_id]['tfidf'][answer_id][sentence_id] \
                        + self.LEXRANK_WEIGHT * self.multi_docs_scores[question_id]['lexrank'][answer_id][sentence_id] \
                        + self.KEYWORDS_WEIGHT * self.multi_docs_scores[question_id]['keywords'][answer_id][sentence_id] \
                        + self.QUERY_BASED_WEIGHT * self.multi_docs_scores[question_id]['query_based'][answer_id][sentence_id]
        # Getting some best sentences
        logging.info('Multi docs summarizing')
        for question_id, question in self.multi_docs.items():
            answer_id = '{}_Answer1'.format(question_id)
            answer = question['answers'][answer_id]
            pairs = []
            for sentence_id, sentence in answer['sentences'].items():
                pairs.append((self.multi_docs_scores[question_id]['final'][answer_id][sentence_id], sentence_id))
            pairs = sorted(pairs, reverse=True)
            if len(pairs) > cutoff:
                pairs = pairs[:cutoff]
            # Reordering to original order
            pairs = sorted(pairs, key=lambda x: x[1])
            summary = ' '.join([answer['sentences'][sentence_id]['sentence'] for score, sentence_id in pairs])
            self.answers[question_id] = summary

    def evaluate(self):
        precision, recall, f1 = [], [], []
        for question_id, summary in self.answers.items():
            predict = self.answers[question_id]
            answer = self.data_loader.docs[question_id]['extractive_summary']
            score = evaluate_rouge(predict, answer)
            precision.append(score['rouge2'].precision)
            recall.append(score['rouge2'].recall)
            f1.append(score['rouge2'].fmeasure)
        macro_precision = sum(precision) / len(precision)
        macro_recall = sum(recall) / len(recall)
        macro_f1 = sum(f1) / len(f1)
        logging.info('Evaluate: {} {} {}'.format(macro_precision, macro_recall, macro_f1))
        return macro_precision, macro_recall, macro_f1

    def export_data(self, path='./data/Test/TestSet_MAS.txt'):
        logging.info('Export extractive summary')
        results = []
        for question_id, question in self.answers.items():
            results.append(question_id + '\t' + question)
        results = '\n'.join(results)
        with open(path, 'w') as file:
            file.write(results)
        logging.info('Export extractive done')
