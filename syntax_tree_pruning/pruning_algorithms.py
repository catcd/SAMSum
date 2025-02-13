import logging
import string

from nltk.parse.corenlp import CoreNLPParser
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.evaluation import evaluate_rouge
from utils.pre_processing import TextCleaner, TextSegmentator, nlp


def skip_tokenize(text):
    return text


class HeuristicalPruning:
    # Tags
    NOUN_LABELS = ['NP']
    VERB_LABELS = ['VP']
    ADJECTIVE_LABELS = ['ADJP']
    ADVERB_LABELS = ['ADVP']
    RELATIVE_LABELS = ['WHNP', 'WHADVP', 'WHPP']
    CONJUNCTION_LABELS = ['CONJP']
    PREPOSITION_LABELS = ['PP']

    # Bonus
    VERB_BONUS = 0.7
    NOUN_BONUS = 0.7

    # Penalty
    ADJECTIVE_PENALTY = 0.6
    ADVERB_PENALTY = 0.5
    RELATIVE_PENALTY = 0.3
    CONJUNCTION_PENALTY = 0.3
    PREPOSITION_PENALTY = 0.8

    # Weight
    TAG_WEIGHT = 0.9
    TFIDF_WEIGHT = 1
    TEXTRANK_WEIGHT = 0.7
    QUESTION_DRIVEN_WEIGHT = 0.5

    # Allow pruning labels
    ALLOW_PRUNING_LABELS = ['ADJP', 'ADVP', 'WHNP', 'WHADVP', 'WHPP', 'PP']

    # Thresholds
    MIN_TOKENS_THRESHOLD = 2
    SCORE_THRESHOLD = 1.2
    SENTENCE_LENGTH_LIMIT = 300

    def __init__(self, data_loader, params=None):
        self.data_loader = data_loader
        self.parser = CoreNLPParser()
        if params:
            if 'VERB_BONUS' in params:
                self.VERB_BONUS = params['VERB_BONUS']
            if 'NOUN_BONUS' in params:
                self.VERB_BONUS = params['NOUN_BONUS']
            if 'ADJECTIVE_PENALTY' in params:
                self.ADJECTIVE_PENALTY = params['ADJECTIVE_PENALTY']
            if 'ADVERB_PENALTY' in params:
                self.VERB_BONUS = params['ADVERB_PENALTY']
            if 'RELATIVE_PENALTY' in params:
                self.VERB_BONUS = params['RELATIVE_PENALTY']
            if 'CONJUNCTION_PENALTY' in params:
                self.CONJUNCTION_PENALTY = params['CONJUNCTION_PENALTY']
            if 'PREPOSITION_PENALTY' in params:
                self.PREPOSITION_PENALTY = params['PREPOSITION_PENALTY']
            if 'TAG_WEIGHT' in params:
                self.TAG_WEIGHT = params['TAG_WEIGHT']
            if 'TFIDF_WEIGHT' in params:
                self.TFIDF_WEIGHT = params['TFIDF_WEIGHT']
            if 'TEXTRANK_WEIGHT' in params:
                self.TEXTRANK_WEIGHT = params['TEXTRANK_WEIGHT']
            if 'QUESTION_DRIVEN_WEIGHT' in params:
                self.QUESTION_DRIVEN_WEIGHT = params['QUESTION_DRIVEN_WEIGHT']
        self.phrase_scores = None

    def preprocess_extractive_summaries(self):
        logging.info('Preprocess extractive summaries')
        for question_id, data in self.data_loader.docs.items():
            text = TextCleaner().clean(data['extractive_summary'])
            text_segmentator = TextSegmentator(text)
            self.data_loader.docs[question_id]['extractive_summary_sentences'] = text_segmentator.tokenize()

    def calculate_tag_score(self, label):
        if label in self.NOUN_LABELS:
            return self.NOUN_BONUS
        if label in self.VERB_LABELS:
            return self.VERB_BONUS
        if label in self.ADJECTIVE_LABELS:
            return -self.ADJECTIVE_PENALTY
        if label in self.ADVERB_LABELS:
            return -self.ADVERB_PENALTY
        if label in self.RELATIVE_LABELS:
            return -self.RELATIVE_PENALTY
        if label in self.CONJUNCTION_LABELS:
            return -self.CONJUNCTION_PENALTY
        if label in self.PREPOSITION_LABELS:
            return -self.PREPOSITION_PENALTY
        return 0

    def calculate_tfidf_score(self, text, docs):
        # Adding corpus
        corpus = []
        for answer_id, answer in docs['answers'].items():
            for sentence_id, sentence in answer['sentences'].items():
                sentence_corpus = []
                for token in sentence['tokens']:
                    if not token['stop'] and token['pos'] != 'PUNCT':
                        sentence_corpus.append(token['lemma'])
                corpus.append(sentence_corpus)
        # Calculating TF-IDF
        vectorizers = TfidfVectorizer(tokenizer=skip_tokenize, lowercase=False)
        token_scores = vectorizers.fit_transform(corpus)
        feature_names = vectorizers.get_feature_names()
        feature_scores = [0] * len(feature_names)
        for token_score in token_scores:
            token_score = token_score.toarray().flatten()
            for i in range(len(feature_names)):
                feature_scores[i] = max(feature_scores[i], token_score[i])
        score = 0
        for i in range(len(feature_names)):
            if feature_names[i] in text:
                score += feature_scores[i]
        return score

    def calculate_textrank_score(self, text, docs):
        doc = nlp(docs['extractive_summary'])
        text_doc = nlp(text)
        score, max_similarity = 0, 0
        for phrase in doc._.phrases:
            phrase_doc = nlp(phrase.text)
            similarity = text_doc.similarity(phrase_doc)
            if max_similarity < similarity:
                score = phrase.rank
                max_similarity = similarity
        return score

    def calculate_question_driven_score(self, text, docs):
        text_doc = nlp(text)
        question_doc = nlp(docs['question'])
        return question_doc.similarity(text_doc)

    def dfs_parse_tree(self, node, data):
        tag_score = self.calculate_tag_score(node._label)
        # Child node
        if type(node[0]) == str:
            texts = []
            for child in node:
                texts.append(child)
            return tag_score, ' '.join(texts)
        # DFS to child nodes
        phrase = ''
        for i in range(len(node)):
            child_tag_score, child_phrase = self.dfs_parse_tree(node[i], data)
            tag_score += child_tag_score
            if len(phrase) == 0 or node[i]._label in string.punctuation:
                phrase += child_phrase
            else:
                phrase += ' ' + child_phrase
        if len(phrase.split(' ')) >= self.MIN_TOKENS_THRESHOLD and node._label in self.ALLOW_PRUNING_LABELS:
            tfidf_score = self.calculate_tfidf_score(phrase, data)
            textrank_score = self.calculate_textrank_score(phrase, data)
            question_driven_score = self.calculate_question_driven_score(phrase, data)
            self.phrase_scores.append((phrase, self.TAG_WEIGHT * tag_score + self.TFIDF_WEIGHT * tfidf_score \
                                       + self.TEXTRANK_WEIGHT * textrank_score + self.QUESTION_DRIVEN_WEIGHT * question_driven_score))
        return tag_score, phrase

    def pruning_sentences(self, export=False, path=None):
        result = ''
        for question_id, data in self.data_loader.docs.items():
            logging.info('Pruning the extractive summary for question {}'.format(question_id))
            shorten_summary = []
            for sentence_id, sentence_data in data['extractive_summary_sentences'].items():
                # Skip too long sentences
                if len(sentence_data['sentence']) > self.SENTENCE_LENGTH_LIMIT:
                    shorten_summary.append(TextCleaner.remove_whitespaces(sentence_data['sentence']))
                    continue
                # DFS syntax tree
                root = next(self.parser.raw_parse(sentence_data['sentence']))
                self.phrase_scores = []
                self.dfs_parse_tree(root, data)
                # Pruning
                shorten_sentence = sentence_data['sentence']
                for phrase, score in self.phrase_scores:
                    if score < self.SCORE_THRESHOLD:
                        shorten_sentence = shorten_sentence.replace(phrase, '')
                shorten_sentence = TextCleaner.remove_whitespaces(shorten_sentence)
                shorten_summary.append(shorten_sentence)
            self.data_loader.docs[question_id]['shorten_summary'] = ' '.join(shorten_summary)
            if export:
                result += question_id + '\t' + self.data_loader.docs[question_id]['shorten_summary'] + '\n'
        if export:
            with open(path, 'w') as file:
                file.write(result)

    def evaluate(self):
        precision, recall, f1, compress_ratios = [], [], [], []
        for key, data in self.data_loader.docs.items():
            extractive_summary = data['shorten_summary']
            abstractive_summary = data['abstractive_summary']
            score = evaluate_rouge(extractive_summary, abstractive_summary)
            precision.append(score['rouge2'].precision)
            recall.append(score['rouge2'].recall)
            f1.append(score['rouge2'].fmeasure)
            compress_ratios.append(len(extractive_summary) / len(data['extractive_summary']))
        macro_precision = sum(precision) / len(precision)
        macro_recall = sum(recall) / len(recall)
        macro_f1 = sum(f1) / len(f1)
        macro_compress_ratio = sum(compress_ratios) / len(compress_ratios)
        return macro_precision, macro_recall, macro_f1, macro_compress_ratio
