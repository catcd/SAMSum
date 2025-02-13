from utils.pre_processing import nlp
from utils.utils import normalize_score


class QueryBasedScorer:
    @staticmethod
    def scoring(docs):
        # Scoring sentences
        scores = {}
        query_doc = nlp(docs['question'])
        for answer_id, answer in docs['answers'].items():
            scores[answer_id] = {}
            for sentence_id, sentence in answer['sentences'].items():
                sentence_doc = nlp(sentence['sentence'])
                scores[answer_id][sentence_id] = sentence_doc.similarity(query_doc)
        scores = normalize_score(scores)
        return scores
