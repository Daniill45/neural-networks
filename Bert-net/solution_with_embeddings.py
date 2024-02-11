# coding: utf8
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostClassifier
import torch
from transformers import AutoTokenizer, AutoModel
import pymorphy2
import re

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")


def embed_bert_cls(df, model, tokenizer):
    full_embeddins = []
    for i in df.index:
        text = df.loc[i, "text"]
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        full_embeddins.append(embeddings[0].cpu().numpy())
    return full_embeddins


def get_pca_kmeans_tfidf():
    objects = []
    with (open("pca.pickle", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    with (open("kmeans.pickle", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    with (open("tfidf.pickle", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects


def get_kmeans_features(df, pca, kmeans):
    embeddings = np.array(embed_bert_cls(df, model, tokenizer))
    centered = embeddings - embeddings.mean()  # центрируем
    dists_columns = [f'DistanceToCluster_{i}' for i in range(20)]  # создаем названия колонкам
    pca_decomp = pca.transform(centered)
    dists_df = pd.DataFrame(
        data=kmeans.transform(pca_decomp),
        columns=dists_columns
    )
    df = pd.concat([df.reset_index(), dists_df], axis=1)
    return df


morph = pymorphy2.MorphAnalyzer()


def clean_text(data):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        data = data.replace(p, f' {p} ')
    data = re.sub(r'\n', ' ', data)
    data = re.sub('  ', " ", data)
    data = data.lower()
    data = re.sub(r'[0-9.,!?\'\:)("]', ' ', data)
    data = re.sub('[^\x00-\x7Fа-яА-Я]', '', data)
    data = re.sub('  ', " ", data)
    return data


def lemmatize(text):
    words = text.split()  # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return " ".join(res)


new_columns = ['оператор',
               'поддержка',
               'обращение',
               'продавец',
               'консультант',
               'очередь',
               'персонал',
               'вежливость',
               'сотрудник',
               'касса',
               'компетентность',
               'время']


def create_columns(df):
    df["lenght"] = df.text.apply(lambda x: len(x))
    df[new_columns] = 0
    for i in df.index:
        s = df.loc[i].lem_text
        col_to_update = list(set(s.split()).intersection(set(new_columns)))
        df.loc[i, col_to_update] = 1


def get_model():
    model_path = "catboost_model_m_tech"
    from_file = CatBoostClassifier()
    loaded_model = from_file.load_model(model_path)
    return loaded_model


outputcolumns = ['lenght', 'оператор', 'поддержка', 'обращение', 'продавец',
                 'консультант', 'очередь', 'персонал', 'вежливость', 'сотрудник',
                 'касса', 'компетентность', 'время', 'DistanceToCluster_0',
                 'DistanceToCluster_1', 'DistanceToCluster_2', 'DistanceToCluster_3',
                 'DistanceToCluster_4', 'DistanceToCluster_5', 'DistanceToCluster_6',
                 'DistanceToCluster_7', 'DistanceToCluster_8', 'DistanceToCluster_9',
                 'DistanceToCluster_10', 'DistanceToCluster_11', 'DistanceToCluster_12',
                 'DistanceToCluster_13', 'DistanceToCluster_14', 'DistanceToCluster_15',
                 'DistanceToCluster_16', 'DistanceToCluster_17', 'DistanceToCluster_18',
                 'DistanceToCluster_19', 'TotalTfIdf', 'MaxTfIdf', 'MeanTfIdf']
d = {0: 'Консультация КЦ',
     1: 'Компетентность продавцов/ консультантов',
     2: 'Электронная очередь',
     3: 'Доступность персонала в магазине',
     4: 'Вежливость сотрудников магазина',
     5: 'Обслуживание на кассе',
     6: 'Обслуживание продавцами/ консультантами',
     7: 'Время ожидания у кассы'}


def get_result(text: pd.Series):
    df = pd.DataFrame(text, columns=["text"])
    pca, kmeans, tfidf = get_pca_kmeans_tfidf()
    df = get_kmeans_features(df, pca, kmeans)
    df.loc[:, "clean_text"] = df.text.apply(lambda x: clean_text(x))
    df.loc[:, "lem_text"] = df.clean_text.apply(lambda x: lemmatize(x))
    create_columns(df)
    tfidf_test = tfidf.transform(df["lem_text"]).toarray()
    df['TotalTfIdf'] = tfidf_test.sum(axis=1)
    df['MaxTfIdf'] = tfidf_test.max(axis=1)
    df['MeanTfIdf'] = tfidf_test.mean(axis=1)
    df = df.drop(["clean_text", "lem_text", "text", "index"], axis=1)
    df = df[outputcolumns]
    catboost = get_model()
    res = catboost.predict(df)
    return pd.DataFrame([d[i[0]] for i in res], columns=["class_predicted"])
