import wget

try:
    f = open('src/navec_hudlit_v1_12B_500K_300d_100q.tar')
    f.close()
except IOError:
    url = 'https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar'
    wget.download(url, 'src/navec_hudlit_v1_12B_500K_300d_100q.tar')

from natasha import Doc, MorphVocab, Segmenter, NewsMorphTagger, NewsSyntaxParser, NewsEmbedding
from navec import Navec
navec = Navec.load('src/navec_hudlit_v1_12B_500K_300d_100q.tar')

import re

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
from numpy.linalg import norm

import pandas as pd

from transformers import T5ForConditionalGeneration, T5Tokenizer


##########


# стоп-лист русских и английских слов
stop_voc = set()

# загрузка английских стоп-слов
with open('src/stopwords-en.txt', 'r', encoding='utf-8') as file:
    stop_voc = stop_voc | set(file.read().split('\n'))

# загрузка русских стоп-слов
with open('src/stopwords-ru.txt', 'r', encoding='utf-8') as file:
    stop_voc = stop_voc | set(file.read().split('\n'))


def text_stopwords_rem(text):
    # функция для удаления русских и английских стоп-слов,
    # ввод-вывод - текст, стоп-слова в множестве stop_voc
    out_text = []

    for word in text.split():
           if (word not in stop_voc) & (word[:-1] not in stop_voc):
               out_text.append(word)

    return ' '.join(out_text)


##########


def text_cleaner(input_text):
    """Функция для очистки текста. Оставляются точки,
    русские и английские слова, цифры и символы, одиночные
    пробелы и заглавные буквы. Используется для очистки
    текста перед эмбеддингом Natasha (navec), где используются
    точки, заглавные буквы и т.д. Если после очистки остается
    один пробел - выводится пустой текст.

    Args:
        input_text (str): текст для очистки (пост и комментарий)

    Returns:
        clean_text (str): текст после очистки (пост и комментарий)
    """

    # если текст пустой
    if input_text == '':
        return ''

    # замена HTML-тегов на пробел
    clean_text = re.sub('<[^<]+?>', ' ', input_text)

    # замена ссылок с URL на пробел
    clean_text = re.sub(r'http\S+', ' ', clean_text)

    # замена эл почты на пробел
    clean_text = re.sub('([\w\.\-\_]+@[\w\.\-\_]+)', ' ', clean_text)

    # замена (\n \r \t \f) на пробел
    clean_text = re.sub('[\n\r\t\f]', ' ', clean_text)

    # замена наборов символов в начале на пробел
    clean_text = re.sub('^\W+\s', ' ', clean_text)

    # замена наборов символов на пробел
    clean_text = re.sub(r'\s\W+\s', ' ', clean_text)

    # если текст пустой
    if clean_text == '':
        return ''

    # удаление # (хештега) с символами до пробела
    clean_text = ' '.join([word for word in clean_text.split() if word[0] != '#'])

    # замена ! и ? на точку с пробелом
    clean_text = re.sub('!|\?', '. ', clean_text)

    # замена всего, кроме "слов", цифр, точки и пробела на пробел
    clean_text = re.sub('[^а-яА-ЯёЁa-zA-Z0-9\.\s]', ' ', clean_text)

    # замена пробела и точки на одну точку
    clean_text = re.sub(r'\s\.', '.', clean_text)

    # замена точки между символами на пробел (кроме цифр)
    clean_text = re.sub(r'\b[^\d]\.[\d]\b', ' ', clean_text)

    # удаление всех пробелов в начале и в конце,
    clean_text = re.sub('^\s+|\s+$', '', clean_text)

    # если текст пустой
    if clean_text == '':
        return ''

    # вставка пробела между точкой впереди слова вплотную
    clean_text = ' '.join([word if word[0] != '.' else word[1:] for word in clean_text.split()])

    # удаление одиночного символа с точкой
    clean_text = re.sub(r'\s.\.\s', '', clean_text)

    # замена двух и более пробелов на один пробел
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # замена многоточий на точку
    clean_text = re.sub(r'\.+', '.', clean_text)

    # если только один пробел
    if clean_text == ' ':
        return ''

    return clean_text


##########


def text_cleaner_2(input_text):
    """Функция для очистки текста. Удаление одиночных символов,
    удаление стоп-слов, удаление цифр и символов (кроме русских,
    точек и пробелов). Если после очистки остается один пробел -
    выводится пустой текст.

    Args:
        input_text (str): текст для очистки (пост и комментарий)

    Returns:
        clean_text (str): текст после очистки (пост и комментарий)
    """

    # если текст пустой
    if input_text == '':
        return ''

    # замена всего, кроме русских букв, точек и пробелов на пробел
    clean_text = re.sub('[^а-яА-ЯёЁ\s\.]', ' ', input_text)

    # удаление стоп-слов
    clean_text = text_stopwords_rem(clean_text)

    # удаление одиночных символов
    clean_text = re.sub(r'\s.\s', ' ', clean_text)

    # замена двух и более пробелов на один пробел
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # если текст пустой или один пробел
    if (clean_text == '') | (clean_text == ' '):
        return ''

    # установка точки в конце, если ее нет
    if clean_text[-1] != '.':
        clean_text += '.'

    return clean_text


##########


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)


def word_vec(text):
    """функция для получения матрицы векторов слов документа
    (NumPy массив размером (300, N). Где N - количество векторов
    слов документа (для которых есть соответствие в словаре).
    Двумерный вектор получается обьединением векторов лемм,
    определенных в тексте (к каждой лемме применен эмбеддинг с
    получением вектора) размера (300, ). Если отсутствует текст
    или все слова текста отсутствуют в словаре navec - вывод None.

    Args:
        text (str): текст

    Returns:
        NumPy array: массив размером (300, N) | None
    """

    # если на входе пустая строка или пробел - возврат None
    if (text == '') | (text == ' '):
        return None

    # сегментация (токенизация) текста, и иная обработка
    # (аннотация в формате Universal Dependities)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)

    # лемматизация с использованием полученных данных о тексте
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    # список лемм с удалением точек
    lemm_text = [el for el in [_.lemma for _ in doc.tokens] if el != '.']

    # векторизация лемм и сборка массива 300 x N
    arr_vec = np.empty((300, 0))
    for word in lemm_text:
        if word in navec:
            arr_vec = np.append(arr_vec, navec[word].reshape(300, 1), axis=1)

    # если нет ни одного вектора леммы ("пустой" текст) - возврат None
    if arr_vec.shape[1] == 0:
        return None

    return arr_vec


##########


def doc_vec_avr(text):
    """Функция для получения вектора документа (NumPy массив
    размером (300, )) из двумерного массива векторов (обьединение
    векторов лемм, определенных в тексте. Используются 5 методов
    понижения размерности: PCA, KernelPCA, TruncatedSVD, Isomap и
    SpectralEmbedding и вычисляется среднее (вектор усредненных значений).

    Args:
        text (str): текст

    Returns:
        NumPy array: массив размером (300, ) | None
    """

    # если на входе пустая строка или пробел - возврат None
    if (text == '') | (text == ' '):
        return None

    # получение массива векторов для текста
    arr_vec = word_vec(text)

    # если нет ни одного вектора для лемм - возврат None
    if arr_vec is None:
        return None

    # если размерность вектора документа (300, 1), т.е.
    # в тексте только одна лемма из списка navec
    if arr_vec.shape[1] == 1:
        return arr_vec.reshape(300, )

    # понижение размерности (сведение набора векторов лемм
    # всего текста к одному вектору размерности (300, ))
    # с помощью 5 методов по отдельности
    pca = PCA(n_components=1).fit_transform(arr_vec).reshape(300, )
    kpca = KernelPCA(n_components=1, n_jobs=-1, random_state=42).fit_transform(arr_vec).reshape(300, )
    tsvd = TruncatedSVD(n_components=1, random_state=42).fit_transform(arr_vec).reshape(300, )
    isomap = Isomap(n_neighbors=5, n_components=1, n_jobs=-1).fit_transform(arr_vec).reshape(300, )
    spemb = SpectralEmbedding(n_components=1).fit_transform(arr_vec).reshape(300, )

    # вычисление усредненного вектора из 5 векторов,
    # полученных методами понижения размерности

    return (pca + kpca + tsvd + isomap + spemb) / 5


##########


def doc_vec_sml(text):
    """Функция для получения вектора документа (NumPy
    массив размером (300, )) из двумерного массива векторов
    (обьединение векторов лемм, определенных в тексте.
    Используется усреднение векторов лемм.

    Args:
        text (str): текст

    Returns:
        NumPy array: массив размером (300, ) | None
    """

    # если на входе пустая строка или пробел - возврат None
    if (text == '') | (text == ' '):
        return None

    # получение массива векторов для текста
    arr_vec = word_vec(text)

    # если нет ни одного вектора для лемм - возврат None
    if arr_vec is None:
        return None

    # если размерность вектора документа (300, 1), т.е.
    # в тексте только одна лемма из списка navec
    if arr_vec.shape[1] == 1:
        return arr_vec.reshape(300, )

    return np.mean(arr_vec, axis=1)


##########


def cosine(comm_vec, post_vec):
    # функция вычисления косинусной близости
    # между двумя векторами, аргументы -
    # вектор комментария и вектор поста,
    # если хотя бы одного не существует, вывод None
    if (comm_vec is not None) & (post_vec is not None):
        return np.dot(comm_vec, post_vec)/(norm(comm_vec)*norm(post_vec))
    return None


##########


def vec_mean(series_vec):
    # функция для вычисления "среднего" вектора
    # из набора векторов размерности (300, ) из
    # Series DataFrame. Использование NumPy.mean().
    ln = series_vec.shape[0]
    arr = np.empty((300, 0))
    for i in range(ln):
        arr = np.append(arr, series_vec[i].reshape(300, 1), axis=1)

    return np.mean(arr, axis=1).reshape(300, )


##########


def determine_k(series_vec):
    """Функция для определения оптимального количества
    кластеров для кластеризации набора векторов документов
    размерности (300, ). Используется коэффициент силуэта и
    K-Means.
    Если во входящей сери меньше 2 элементов, вывод - None.

    Args:
        series_vec(pandas.Series): серия векторов документов

    Returns:
        k(int): оптимальное число кластеров | None
    """

    # проверка количества элементов
    if series_vec.shape[0] < 2:
        return None

    # массив векторов документов (N, 300),
    # где N количество документов (комментариев )
    ln = series_vec.shape[0]
    arr = np.empty((300, 0))

    for i in range(ln):
        arr = np.append(arr, series_vec[i].reshape(300, 1), axis=1)

    arr = arr.T

    # определение коэффициентов силуэта для количества
    # кластеров от 2 до количествва векторов документов
    best_stamp = []

    for i in range(2, min(ln, 10)):

        model = KMeans(n_clusters = i, n_init='auto', random_state=42)
        model.fit(arr)

        best_stamp.append(silhouette_score(arr, model.labels_, metric='euclidean'))

    # количество кластеров при максимальном
    # значении коэффициента силуэта
    k = best_stamp.index(max(best_stamp)) + 2

    return k


##########


def doc_clustering(series_vec, k):
    """Функция для кластерpизации набора векторов документов
    размерности (300, ). Используется K-Means.

    Args:
        series_vec(pandas.Series): серия векторов документов
        k(int): количество кластеров

    Returns:
        NumPy array: метки (1-мерный)
        NumPy array: центры кластеров (2-мерный, метка кластера х вектор центра)
    """

    # массив векторов документов (N, 300),
    # где N количество документов (комментариев )
    ln = series_vec.shape[0]
    arr = np.empty((300, 0))

    for i in range(ln):
        arr = np.append(arr, series_vec[i].reshape(300, 1), axis=1)

    arr = arr.T

    model = KMeans(n_clusters = k, n_init='auto', max_iter=500, random_state=42)
    model.fit(arr)

    return model.labels_, model.cluster_centers_


##########


def select_comments(text, hash, comments_, type_summ):
    """функция для отбора комментариев для конкретного
    поста. Очистка текстов комментариев, удаление из них
    спама, разделение комментариев к посту или к топику
    поста (при необходимости). Удаление спама - косинусная
    близость векторного представления текста комментария и
    "усредненного" вектора текстов комментариев. Разделение
    на две группы - кластеризация (K-Means).

    Args:
        text (str): текст поста
        hash (str): хеш поста
        comments_: DataFrame (комментарии к данному посту и хеш)
        type_summ: тип суммирования

    Returns:
        dict: словарь с тремя элементами
        (summary - отобранные комментарии списком или
        пустой список, post_hash - хеш поста и
        comments_hash - список хешей комментариев или пустой список)
    """

    # "пустой" словарь
    empty_res = {'summary': [],
                 'post_hash': [hash],
                 'comments_hash': []}

    # очистка текста комментариев
    comments_['clean_text'] = comments_['text'].apply(text_cleaner)
    comments_ = comments_.reset_index(drop=True)

    # если комментариев вообще нет (после очистки
    # текста) - возврат пустого словаря
    comments_ = comments_[comments_['clean_text'] != '']
    if comments_.shape[0] == 0:
        return empty_res

    #
    ### удаление возможного СПАМА в комментариях
    #

    # векторы документов комментариев (усредненные
    # векторы из векторов слов текстов комментариев)
    comments_['doc_vec_sml'] = comments_['clean_text'].apply(doc_vec_sml)

    # удаление комментариев с пустыми
    # векторами документов комментариев (где None)
    mask = comments_['doc_vec_sml'] == comments_['doc_vec_sml']
    comments_ = comments_[mask].reset_index(drop=True)

    # если нет комментариев (после векторизации
    # документов комментариев) - возврат пустого словаря
    if comments_.shape[0] == 0:
        return empty_res

    # "средний" вектор документов комментариев
    comm_doc_vec_sml = vec_mean(comments_['doc_vec_sml'])

    # косинусная близость между векторами документов комментариев и
    # "средним" вектором документов комментариев
    comments_['cosine_sml'] = comments_['doc_vec_sml'].apply(cosine,
                                                             args=(comm_doc_vec_sml,))

    # порог по косинусной близости 0.4 (меньше - удаление)
    comments_ = comments_[comments_['cosine_sml'] >= 0.4]

    # если нет комментариев (после удаления СПАМа) -
    # возврат пустого словаря
    if comments_.shape[0] == 0:
        return empty_res


    #
    ### разделение комментариев на комментарии к посту и к топику поста
    #


    # векторы документов комментариев, 5 методов понижения
    # размерности для получения из массивов векторов слов
    # документов комментариев - векторов документов комментариев
    comments_['doc_vec_avr'] = comments_['clean_text'].apply(doc_vec_avr)

    # удаление комментариев с пустыми
    # векторами документов комментариев (где None)
    mask = comments_['doc_vec_avr'] == comments_['doc_vec_avr']
    comments_ = comments_[mask].reset_index(drop=True)

    # если нет комментариев (после векторизации
    # документов комментариев) - возврат пустого словаря
    if comments_.shape[0] == 0:
        return empty_res

    # словарь со всеми комментариями
    all_res = {'summary': comments_['clean_text'].to_list(),
               'post_hash': [hash],
               'comments_hash': comments_['hash'].to_list()}

    # вектор документа поста
    post_doc_vec_text = doc_vec_avr(text_cleaner(text))

    # возврат всех имеющихся комментариев в случаях:
    # если тип суммаризации all_comments
    if type_summ == 'all_comments':
        return all_res
    # если вектор документа поста не существует и тип
    # суммаризации post_comments
    elif (post_doc_vec_text is None) & (type_summ == 'post_comments'):
        return all_res
    # если вектор документа поста не существует и тип
    # суммаризации topic_comments
    elif (post_doc_vec_text is None) & (type_summ == 'topic_comments'):
        return all_res

    # тип суммаризации post_comments или topic_comments и
    # только один комментарий (порог по косинусной близости 0.6)
    if comments_.shape[0] == 1:
        cos_dist = cosine(comments_['doc_vec_avr'][0], post_doc_vec_text)
        # если тип суммаризации post_comments и косинусная
        # близость больше 0.6
        if (type_summ == 'post_comments') & (cos_dist > 0.6):
            return all_res
        # если тип суммаризации topic_comments и косинусная
        # близость не больше 0.6
        elif (type_summ == 'topic_comments') & (cos_dist <= 0.6):
            return all_res
        # два иных случая - возврат пустого словаря
        else:
            return empty_res

    # бинарная кластеризация массива векторов
    # документов комментариев
    labels, cluster_centers = doc_clustering(comments_['doc_vec_avr'], 2)

    # создание признака меток для комментариев
    comments_['label'] = labels

    # вычисление косинусной близости между центром кластера
    # и вектором документа поста, определение ее максимума и
    # определение кластера комментариев к посту и кластера
    # комментариев к топику
    cosine_class_0 = cosine(cluster_centers[0], post_doc_vec_text)
    cosine_class_1 = cosine(cluster_centers[1], post_doc_vec_text)
    # вывод комментариев нужного типа (зависит от косинусной близости
    # центра кластера к вектору документа поста: тот кластер, чей центр
    # ближе к вектору документа поста - кластер комментариев к посту, а
    # тот кластер, чей центр дальше - кластер комментариев к топику)
    if cosine_class_0 >= cosine_class_1:
        if type_summ == 'post_comments':
            return {'summary': comments_.loc[comments_['label'] == 0, 'clean_text'].to_list(),
                    'post_hash': [hash],
                    'comments_hash': comments_.loc[comments_['label'] == 0, 'hash'].to_list()}
        elif type_summ == 'topic_comments':
            return {'summary': comments_.loc[comments_['label'] == 1, 'clean_text'].to_list(),
                    'post_hash': [hash],
                    'comments_hash': comments_.loc[comments_['label'] == 1, 'hash'].to_list()}
    else:
        if type_summ == 'post_comments':
            return {'summary': comments_.loc[comments_['label'] == 1, 'clean_text'].to_list(),
                    'post_hash': [hash],
                    'comments_hash': comments_.loc[comments_['label'] == 1, 'hash'].to_list()}
        elif type_summ == 'topic_comments':
            return {'summary': comments_.loc[comments_['label'] == 0, 'clean_text'].to_list(),
                    'post_hash': [hash],
                    'comments_hash': comments_.loc[comments_['label'] == 0, 'hash'].to_list()}


##########


MODEL_NAME = 'cointegrated/rut5-base-absum'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


def summary_making(text, min_length=30):
    """функция суммаризации. Используется модель, основанная
    на Т5. Текст подготовлен, в виде единой строки, присутствуют
    точки и заглавные буквы. Входной параметр - минимальная длина,
    используемая по умолчанию в методе генерации модели (необходимо
    определять исходя из размера текста).

    Args:
        text (str): текст для суммаризации.
        min_length (int, optional): минимальное количество слов,
        требующеесчя в саммари (по умолчанию 40).

    Returns:
        (str): текст саммари.
    """

    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt")
    summary_ids = model.generate(
        input_ids,
        num_beams = 4,
        no_repeat_ngram_size = 2,
        min_length = min_length,
        max_length = 128,
        repetition_penalty=10.0,
        length_penalty=2.0,
        early_stopping=True,
        n_words=None,
        compression=None,
        do_sample=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


##########


def summarization(list_text):
    """функция для обработки набора коментариев и получения
    текста их общей суммаризации. Принимаемые комментарии
    кластеризуются автоматически (K-means и коэффициент силуэта
    для определения оптимального количества кластеров).
    Суммаризация для каждого кластера отдельно.
    Большое количество разнородных комментариев нежелательно.

    Args:
        list_text (list): список комментариев для суммаризации

    Returns:
        str: текст суммаризации
    """

    # пустой саммари
    empty_summary = ''

    # DataFrame комментариев
    df_comm = pd.DataFrame({'text': list_text})

    # дополнительная очистка текста комментариев
    df_comm['clean_text'] = df_comm['text'].apply(text_cleaner_2)

    # если комментариев вообще нет (после очистки
    # текста) - возврат пустого списка
    df_comm = df_comm[df_comm['clean_text'] != '']
    if df_comm.shape[0] == 0:
        return empty_summary

    # векторы документов комментариев, 5 методов понижения
    # размерности для получения из массивов векторов слов
    # документов комментариев - векторов документов комментариев
    df_comm['doc_vec_avr'] = df_comm['clean_text'].apply(doc_vec_avr)

    # удаление комментариев с пустыми
    # векторами документов комментариев (где None)
    mask = df_comm['doc_vec_avr'] == df_comm['doc_vec_avr']
    df_comm = df_comm[mask].reset_index(drop=True)

    # если нет комментариев (после векторизации
    # документов комментариев) - возврат пустого списка
    if df_comm.shape[0] == 0:
        return empty_summary

    # если не более 5 комментариев, применим к каждому
    # комментарию функцию суммаризации текста
    if df_comm.shape[0] < 6:
        summary = []
        for i in range(df_comm.shape[0]):
            summary.append(summary_making(df_comm['clean_text'][i], min_length=1))
        # и выведем текст из их результатов
        return '\n'.join(summary)


    #
    ### кластеризация
    #


    # определение оптимального количества кластеров
    # для комментариев
    k_num = determine_k(df_comm['doc_vec_avr'])

    # кластеризация комментариев
    labels, cluster_centers = doc_clustering(df_comm['doc_vec_avr'], k_num)

    # создание признака меток для комментариев
    df_comm['label'] = labels

    # применение функции суммаризации для каждого кластера
    # комментариев по отдельности, с выбором не более пяти
    # комментариев, вектора документов которых максимально близки
    # по косинусной близости к центру своего кластера
    summary = []
    for i in range(k_num):
        # список комментариев кластера
        group_comm = df_comm.loc[df_comm['label'] == i]
        # косинусная близость векторов документов
        # комментариев к центру своего кластера
        group_comm['cosine_center'] = group_comm['doc_vec_avr'].apply(cosine, args=(cluster_centers[i],))
        # сортировка по убыванию косинусной близости
        group_comm = group_comm.sort_values(by='cosine_center', ascending=False)
        # выбор не более пяти ближайших комментариев по
        # косинусной близости к центру своего кластера
        best_comm = group_comm['clean_text'].head(min(group_comm.shape[0], 5)).to_list()
        text = ' '.join(best_comm)
        # суммаризация
        summary.append(summary_making(text, min(group_comm.shape[0] * 5, 30)))

    # вывод текстов саммари всех кластеров
    return '\n'.join(summary)
