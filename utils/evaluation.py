import bert_score
from rouge_score import rouge_scorer


def evaluate_rouge(predict, answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(answer, predict)


def evaluate_bert_score(predict, answer):
    return bert_score.score(predict, answer, model_type='xlnet-base-cased', batch_size=16, return_hash=True, verbose=True)
