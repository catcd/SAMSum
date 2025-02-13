import re

import spacy
import pytextrank

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('textrank')


class TextCleaner:
    @staticmethod
    def remove_html_tags(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    @staticmethod
    def remove_whitespaces(text):
        return " ".join(text.split()).lstrip().rstrip()

    def clean(self, text):
        steps = [self.remove_html_tags, self.remove_whitespaces]
        for step in steps:
            text = step(text)
        return text


class TextSegmentator:
    def __init__(self, text):
        self.doc = nlp(text)

    def tokenize(self):
        answer = {}
        sentences_cnt = 0
        for sent in self.doc.sents:
            sentence_id = str(sentences_cnt)
            answer[sentence_id] = {}
            answer[sentence_id]['sentence'] = sent.text
            answer[sentence_id]['ners'] = [ent.text for ent in sent.ents]
            answer[sentence_id]['tokens'] = []
            for token in sent:
                answer[sentence_id]['tokens'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'is_alpha': token.is_alpha,
                    'stop': token.is_stop,
                })
            sentences_cnt += 1
        return answer

    def get_ners(self):
        return [ent.lemma_ for ent in self.doc.ents]

    def get_keywords(self):
        keyword_pos = ['NOUN', 'VERB']
        keywords = []
        for token in self.doc:
            if token.pos_ in keyword_pos:
                keywords.append(token.lemma_)
        return keywords
