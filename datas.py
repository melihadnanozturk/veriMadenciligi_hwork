import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

sns.set()


## GET DATA
def get_data(path, fileName):
    os.chdir(path)
    dataset = pd.read_csv(fileName)
    return pd.DataFrame(dataset)


def remove_data_by_rate_and_fill_column_mean(rate, column, data):
    # Belirli bir oranda veriyi null yapalım
    oran = rate
    kolon = column

    # Null değerlerin oluşturulacağı hücre sayısı
    null_sayisi = int(oran * len(data))

    # Null değerlerin rastgele oluşturulacağı hücrelerin indekslerini seçelim
    null_indeksler = np.random.choice(data.index, null_sayisi, replace=False)
    data.loc[null_indeksler, kolon] = np.nan

    data[kolon] = data[kolon].fillna(value=data[kolon].mean())


def remove_data_by_rate_and_fill_column_values(rate, column, data, value):
    oran = rate
    kolon = column
    null_sayisi = int(oran * len(data))
    null_indeksler = np.random.choice(data.index, null_sayisi, replace=False)
    data.loc[null_indeksler, kolon] = np.nan
    data[kolon] = data[kolon].fillna(value=value)
    print(data[kolon])


def encode_data_by_labelEncoded(data):
    label_encode_data = data.copy()

    label_encoder = LabelEncoder()
    label_encode_data["Gender"] = label_encoder.fit_transform(label_encode_data["Gender"])
    label_encode_data["Occupation"] = label_encoder.fit_transform(label_encode_data["Occupation"])
    label_encode_data["BMI Category"] = label_encoder.fit_transform(label_encode_data["BMI Category"])
    label_encode_data["Sleep Disorder"] = label_encoder.fit_transform(label_encode_data["Sleep Disorder"])
    return label_encode_data


def rename_columns(data):
    return data.rename(columns={
        'Gender': '1',
        'Age': '2',
        'Occupation': '3',
        'Sleep Duration': '4',
        'Quality of Sleep': '5',
        'Physical Activity Level': '6',
        'Stress Level': '7',
        'BMI Category': '8',
        'Heart Rate': '9',
        'Daily Steps': '10',
        'Sleep Disorder': '11',
        'P_High': '12',
        'P_Low': '13'})
