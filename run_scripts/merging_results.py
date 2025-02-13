import logging
import os

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)


def merge(folder, output):
    logging.info('Merging files from {} to {}'.format(folder, output))
    results = []
    for filename in os.listdir(folder):
        question_id = filename.split('.')[0]
        if question_id.isdigit():
            with open(os.path.join(folder, filename), 'r') as file:
                summary = file.read()
                results.append((int(question_id), summary))
    results = sorted(results)
    results = '\n'.join([str(question_id) + '\t' + summary for question_id, summary in results])
    with open(output, 'w') as file:
        file.write(results)


if __name__ == '__main__':
    merge('./data/Bart/', './data/Bart/Test_Bart.txt')
