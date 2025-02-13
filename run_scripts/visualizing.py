import json
import logging
import os

import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('..')
logging.getLogger().setLevel(logging.INFO)


def visualizing_labels():
    ratio = {}
    with open('./data/Statistics/Test_Labels.txt') as file:
        data = json.loads(file.read())
        count = data['inside']
        for key, outside_count in data['outside'].items():
            inside_count = data['inside'][key] if key in data['inside'] else 0
            ratio[key] = inside_count / (inside_count + outside_count)
    logging.info('Count: {}'.format(count))
    logging.info('Ratio: {}'.format(ratio))
    # Plot ratio
    x = ['NP', 'VP', 'PP', 'SBAR', 'ADJP', 'ADVP', 'CONJP']
    y = []
    for label in x:
        y.append(ratio[label])
    plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y)
    # plt.show()
    plt.savefig('./data/Visualizations/Labels_Ratio.png')
    # Plot count
    y = []
    for label in x:
        y.append(count[label])
    plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y)
    # plt.show()
    plt.savefig('./data/Visualizations/Labels_Count.png')


if __name__ == '__main__':
    visualizing_labels()
