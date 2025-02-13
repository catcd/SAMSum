from utils.utils import normalize_score


class KeywordsScorer:
    @staticmethod
    def scoring(docs, keyword_boost=0.3, ner_boost=0.5):
        scores = {}
        for answer_id, answer in docs['answers'].items():
            scores[answer_id] = {}
            for sentence_id, sentence in answer['sentences'].items():
                score = 0
                for token in sentence['tokens']:
                    if token['lemma'] in docs['question_keywords']:
                        score += keyword_boost
                    # Ner can be multiple tokens
                    for ner in docs['question_ners']:
                        if token['lemma'].lower() in ner.lower():
                            score += ner_boost
                            break
                scores[answer_id][sentence_id] = score
        scores = normalize_score(scores)
        return scores
