# from gensim.models import Word2Vec
# import re
# from lxml import html

import numpy as np
import pymorphy2
import nltk
# from selenium import webdriver
# import time
# from urllib.parse import urlencode
from gensim import models
import telebot
from urllib.parse import quote
import pickle
# from collections import Counter
from functools import lru_cache
# from annoy import AnnoyIndex
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
import json
import requests

# df = pd.read_csv('/Users/Valeriya/Downloads/root/LABS/thequestion-hack/data/thequestion.csv')
# df = df.dropna()
# df.to_pickle('pickled_df.pickle')

model = models.KeyedVectors.load_word2vec_format('/Volumes/Transcend/web_0_300_20.bin', binary=True)


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

grammar_mapping = {
    'NOUN': '_NOUN',
    'VERB': '_VERB', 'INFN': '_VERB', 'GRND': '_VERB', 'PRTF': '_VERB', 'PRTS': '_VERB',
    'ADJF': '_ADJ', 'ADJS': '_ADJ',
    'ADVB': '_ADV',
    'PRED': '_ADP'
}


morph_analyzer = pymorphy2.MorphAnalyzer()

@lru_cache(10000)
def infinite(word):
    parsed_word = morph_analyzer.parse(word)[0]
    return parsed_word

def get_vector(str):
    words = nltk.word_tokenize(str) #re.sub("[^а-я]", " ", str, flags=re.IGNORECASE | re.UNICODE)
    total_vector = np.zeros(300)
    total_count = 0
    for word in words:
        parsed_word = morph_analyzer.parse(word)[0]
        if parsed_word.tag.POS is None or not parsed_word.tag.POS in grammar_mapping:
            continue
        converted_word = parsed_word.normal_form + grammar_mapping[parsed_word.tag.POS]
        if not converted_word in model.vocab:
            continue
        word_vector = model.word_vec(converted_word)
        total_count += 1
        total_vector += word_vector
    if total_count > 0:
        total_vector /= total_count

    return total_vector
#
X = pickle.load(open( "/Users/Valeriya/Desktop/question_stop_word_dump.pkl", "rb" ))
nbrs = pickle.load(open('/Users/Valeriya/Downloads/neighbors.pickle', 'rb'))
df = pickle.load(open('/Users/Valeriya/Downloads/pickled_df.pickle', 'rb'))

tg_token = '420371977:AAEtl9DfXPR0RCqdlunPcKMnKYSc5I7cC9M'

bot = telebot.TeleBot(tg_token)

@bot.message_handler(commands=['start'])

def handle_start(message):
   bot.send_message(message.chat.id, "Hi!")

def search_questions(message):
    vec = get_vector(str(message))
    distances, indices = nbrs.kneighbors(vec)
    fin = df.iloc[indices[0],1]
    return fin.to_string().split('    ')[1][:36]

@bot.message_handler(content_types=['text'])

def return_similar_question(question):
    # print('proceeded to returning questions')
    main_link = 'https://thequestion.ru/search/questions?limit=20&offset=0&q='
    ques = search_questions(question)
    encoded = quote(ques)
    search_link = main_link+encoded
    res = requests.get(search_link)
    try:
        question_id = json.loads(res.text)['items'][0]['id']
        final_link = 'Вам может быть интересен следующий ответ: http://thequestion.ru/questions/' + str(question_id)
        bot.send_message(question.chat.id, final_link)
    except:
        bot.send_message(question.chat.id, 'Похожих вопросов не найдено')


if __name__ == '__main__':
   bot.polling(none_stop=True)
