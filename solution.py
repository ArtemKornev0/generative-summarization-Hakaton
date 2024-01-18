from sys import argv
import numpy as np
import pandas as pd
import jsonlines

from src.moduls import *


type_summ = argv[1] # тип суммаризации
dataset   = argv[2] # путь и название файла с данными
result    = argv[3] # путь и название файла с результатом


# получение DataFrame c постами
# получение DataFrame с комментариями
post = []
comment = []

with jsonlines.open(dataset) as reader:
    for obj in reader:
        if obj.get('root_id') is None:
            post.append(obj)
        else:
            comment.append(obj)

df_post = pd.DataFrame(post)
post_del = ['url', 'date']
df_post = df_post.drop(post_del, axis=1) # удаление лишнего

df_comment = pd.DataFrame(comment)
comment_del = ['url', 'id', 'parent_id', 'date']
df_comment = df_comment.drop(comment_del, axis=1) # удаление лишнего


# удаление пропусков
df_post    = df_post.dropna(subset=['id', 'hash'])
df_comment = df_comment.dropna(subset=['root_id', 'hash'])


# удаление дубликатов
df_post    = df_post.drop_duplicates(subset='hash', ignore_index=True)
df_comment = df_comment.drop_duplicates(subset='hash', ignore_index=True)


# упорядочение индексов
df_post    = df_post.reset_index(drop=True)
df_comment = df_comment.reset_index(drop=True)

# суммаризация комментариев для каждого поста и
# запись словарей с саммари и хешами в файл
with jsonlines.open(result, mode='w') as writer:
    #  для каждого поста
    for i in range(df_post.shape[0]):
        # подбор комментариев, в зависимости
        # от типа суммаризации
        post_item = select_comments(df_post.iloc[i, 0],
                                    df_post.iloc[i, 2],
                                    # df_comment,
                                    df_comment.loc[df_comment['root_id'] == df_post.iloc[i, 1], ['text', 'hash']],
                                    type_summ)
        # суммаризация отобранных комментариев
        if post_item['summary']:
            post_item['summary'] = summarization(post_item['summary'])
        # если после обработки или эмбеддинга отобранных
        # комментариев остается пустой текст, обнуляется
        # список хешей комментариев
        if post_item['summary'] == '':
            post_item['comments_hash'] = []

        writer.write(post_item)
