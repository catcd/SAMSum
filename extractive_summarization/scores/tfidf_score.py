from sklearn.feature_extraction.text import TfidfVectorizer

from utils.utils import normalize_score


def skip_tokenize(text):
    return text


class TfidfScorer:
    @staticmethod
    def scoring(docs, threshold=0.2):
        # Adding corpus
        corpus = []
        for answer_id, answer in docs['answers'].items():
            for sentence_id, sentence in answer['sentences'].items():
                sentence_corpus = []
                for token in sentence['tokens']:
                    if not token['stop'] and token['pos'] != 'PUNCT':
                        sentence_corpus.append(token['lemma'])
                corpus.append(sentence_corpus)
        # Scoring by tfidf
        vectorizers = TfidfVectorizer(tokenizer=skip_tokenize, lowercase=False)
        token_scores = vectorizers.fit_transform(corpus)
        sentence_scores = []
        for token_score in token_scores:
            score = sum([value for value in token_score.data if value >= threshold])
            sentence_scores.append(score)
        # Matching with original data
        scores = {}
        sentences_count = 0
        for answer_id, answer in docs['answers'].items():
            scores[answer_id] = {}
            for sentence_id, sentence in answer['sentences'].items():
                scores[answer_id][sentence_id] = sentence_scores[sentences_count]
                sentences_count += 1
        scores = normalize_score(scores)
        return scores
