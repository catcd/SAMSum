from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS

from utils.utils import normalize_score


class LexrankScorer:
    @staticmethod
    def scoring(docs, threshold=0.3):
        # Adding documents as corpus
        documents = []
        for answer_id, answer in docs['answers'].items():
            sentences = []
            for sentence_id, sentence in answer['sentences'].items():
                sentences.append(sentence['sentence'])
            documents.append(sentences)
        lxr = LexRank(documents, stopwords=STOPWORDS['en'])
        # Scoring sentences
        scores = {}
        for answer_id, answer in docs['answers'].items():
            sentences = []
            for sentence_id, sentence in answer['sentences'].items():
                sentences.append(sentence['sentence'])
            sentence_scores = lxr.rank_sentences(sentences, threshold=threshold, fast_power_method=True)
            # Matching scores with original data
            scores[answer_id] = {}
            sentences_count = 0
            for sentence_id, sentence in answer['sentences'].items():
                scores[answer_id][sentence_id] = sentence_scores[sentences_count]
                sentences_count += 1
        scores = normalize_score(scores)
        return scores
