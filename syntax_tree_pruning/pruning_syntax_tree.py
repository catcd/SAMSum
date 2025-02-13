import logging
import math
import os
import random

from syntax_tree_pruning.pruning_algorithms import HeuristicalPruning, GBPruning
from utils.data_loader import DataLoader

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)


def heuristical_syntax_tree_pruning():
    logging.info('Heuristical syntax tree pruning')
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    pruning = HeuristicalPruning(data_loader)
    pruning.preprocess_extractive_summaries()
    pruning.pruning_sentences(export=True, path='./data/SyntaxTreePruning/Test_Pruning_Heuristic.txt')
    logging.info('Evaluation: {}'.format(pruning.evaluate()))


def sampling_params(sampling_range):
    # Sampling in range for 7 tags
    result = []
    for i in range(7):
        result.append(round(random.uniform(sampling_range[i][0], sampling_range[i][1]), 1))
    # Freely sampling for 4 weights
    for i in range(4):
        result.append(round(random.uniform(0, 5), 1))
    return result


def statistical_syntax_tree_pruning():
    logging.info('Statistical syntax tree pruning')
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')

    sampling_count = 10
    climbing_count = 100
    learning_rate = 0.1
    cooling_rate = 0.95
    temperature = 5

    # 7 tags + 4 weights
    best_params, best_f1 = [0] * 11, 0
    # Sampling range for 7 tags only
    sampling_range = [[0.6, 1], [0.6, 1], [0.4, 0.8], [0.3, 0.7], [0.2, 0.6], [0.2, 0.6], [0.4, 0.8]]
    for sampling in range(sampling_count):
        # Sampling
        current_params = sampling_params(sampling_range)
        current_f1 = 0
        for loop in range(climbing_count):
            # Hill-climbing
            neighbor_params = current_params
            param_id = random.randint(0, len(current_params) - 1)
            sign = random.randint(0, 1)
            if sign == 0:
                sign = -1
            neighbor_params[param_id] += sign * learning_rate
            # Re-evaluate
            data_loader = DataLoader()
            data_loader.import_data()
            pruning = HeuristicalPruning(data_loader, params={
                'VERB_BONUS': neighbor_params[0], 'NOUN_BONUS': neighbor_params[1],
                'ADJECTIVE_PENALTY': neighbor_params[2], 'ADVERB_PENALTY': neighbor_params[3],
                'RELATIVE_PENALTY': neighbor_params[4], 'CONJUNCTION_PENALTY': neighbor_params[5],
                'PREPOSITION_PENALTY': neighbor_params[6], 'TAG_WEIGHT': neighbor_params[7],
                'TFIDF_WEIGHT': neighbor_params[8], 'TEXTRANK_WEIGHT': neighbor_params[9],
                'QUESTION_DRIVEN_WEIGHT': neighbor_params[10]
            })
            pruning.preprocess_extractive_summaries()
            pruning.pruning_sentences()
            macro_precision, macro_recall, macro_f1, _ = pruning.evaluate()
            logging.warning('Current evaluate: {} {} {} {}'.format(macro_precision, macro_recall, macro_f1, neighbor_params))
            if best_f1 < macro_f1:
                best_f1, best_params = macro_f1, neighbor_params
            # Simulated annealing
            if current_f1 < macro_f1:
                current_f1, current_params = macro_f1, neighbor_params
            else:
                delta = macro_f1 - best_f1
                random_value = random.uniform(0, 1)
                if random_value > math.e ** (-delta / temperature):
                    current_f1, current_params = macro_f1, neighbor_params
            temperature *= cooling_rate
    logging.warning('Best params: {} {}'.format(best_f1, best_params))

    # Export result of best params
    pruning = HeuristicalPruning(data_loader, params={
        'VERB_BONUS': best_params[0], 'NOUN_BONUS': best_params[1],
        'ADJECTIVE_PENALTY': best_params[2], 'ADVERB_PENALTY': best_params[3],
        'RELATIVE_PENALTY': best_params[4], 'CONJUNCTION_PENALTY': best_params[5],
        'PREPOSITION_PENALTY': best_params[6], 'TAG_WEIGHT': best_params[7],
        'TFIDF_WEIGHT': best_params[8], 'TEXTRANK_WEIGHT': best_params[9],
        'QUESTION_DRIVEN_WEIGHT': best_params[10]
    })
    pruning.preprocess_extractive_summaries()
    pruning.pruning_sentences(export=True, path='./data/SyntaxTreePruning/Test_Pruning_Heuristic.txt')
    logging.info('Evaluation: {}'.format(pruning.evaluate()))


def gb_syntax_tree_pruning():
    logging.info('GB syntax tree pruning')
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    pruning = GBPruning(data_loader)
    pruning.preprocess_extractive_summaries()
    pruning.pruning_sentences(export=True, path='./data/SyntaxTreePruning/Test_Pruning_GB.txt')
    logging.info('Evaluation: {}'.format(pruning.evaluate()))


def gb_syntax_tree_pruning_with_bart():
    logging.info('GB syntax tree pruning with BART')
    data_loader = DataLoader()
    data_loader.import_data('./data/Preprocessing/Test_Preprocessing.txt')
    data_loader.load_extractive_summaries('./data/Bart/Test_Bart.txt', delimiter='\t')
    pruning = GBPruning(data_loader)
    pruning.preprocess_extractive_summaries()
    pruning.pruning_sentences(export=True, path='./data/SyntaxTreePruning/Test_Pruning_Bart_GB.txt')
    logging.info('Evaluation: {}'.format(pruning.evaluate()))


if __name__ == '__main__':
    heuristical_syntax_tree_pruning()
    # statistical_syntax_tree_pruning()
    # gb_syntax_tree_pruning()
    # gb_syntax_tree_pruning_with_bart()
