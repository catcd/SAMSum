def normalize_score(scores):
    for answer_id, answer in scores.items():
        min_val, max_val = min(answer.values()), max(answer.values())
        if min_val < max_val:
            for sentence_id, score in answer.items():
                scores[answer_id][sentence_id] = (score - min_val) / (max_val - min_val)
    return scores
