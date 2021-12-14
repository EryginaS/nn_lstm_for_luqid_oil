# -*- coding: utf-8 -*-
"""
В модуле описана модель и процесс обучения модели.
Итоговая модель принимает на вход api скважены,
и предсказывает на 12 месяцев вперед производство жидкой нефти.
Все предсказанные данные были изображены на графиках  для каждой скважины,
лежат в result
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from lstm_model import LSTM


PATH = os.path.join('../data', 'production.csv')
data = pd.read_csv(PATH)
data.head()

"""
Построим графики серий производства (месяц и количество произведенной нейфти)

"""

APIs = data['API'].unique()
plt.title('Production series')
plt.ylabel('Производство')
plt.xlabel('месяц')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(data[data['API'] == APIs[0]]['Liquid'])
plt.savefig('../series_api.png')
dict_ = {}
for api in APIs:
    dict_[api] = list(data[data['API'] == api]['Liquid'])

df = pd.DataFrame(dict_)
df.head()

"""# Предварительная обработка данных и создание обучающих данных"""

train_window = 1

def create_inout_sequences(input_data, tw = 12):
    '''
    Делит данные на обучающие и те, которые будут предсказаны
    in: input_data - input data (list)
        tw - train_window (int)
    out: tuple - 1st elem contains 12 records, 2nd contains 1 record
    '''

    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))

    return inout_seq

from sklearn.preprocessing import MinMaxScaler

train_size = 12     ## размер обучающей выборки

all = {}            ## данные
train_data = {}     ## train data (обучающие данные)
train_data_normalized = {}  ## нормалицованные данные с помощью MinMaxScaler
train_inout_seq = {}        ## последовательность для обучения
scaler = {}                 ## маштаб для каждой api

for i, api in enumerate(APIs):

    all[api] = df[api].values.astype(float)
    train_data[api] = all[api][:train_size]

    scaler[api] = MinMaxScaler(feature_range=(-1, 1))

    train_data_normalized[api] = scaler[api].fit_transform(train_data[api].reshape(-1, 1))
    train_data_normalized[api] = torch.FloatTensor(train_data_normalized[api]).view(-1)

    train_inout_seq[api] = create_inout_sequences(train_data_normalized[api], train_window)

pd.DataFrame(train_data_normalized)

"""модель"""

model = LSTM()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

"""Training"""

epochs = 82

model.train()   ## режим обучения
for api in APIs:    ## для всех api
    print("{} / {}".format(np.where(APIs == api)[0][0], len(APIs)))

    for i in range(epochs):
        for sequence, labels in train_inout_seq[api]:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(6, 1, model.hidden_layer_size),
                            torch.zeros(6, 1, model.hidden_layer_size))

            y_pred = model(sequence)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

"""###### Evaluation"""

fut_pred = 12
test_inputs = {}
for api in APIs:        ## создадим тестовую выборку
    test_inputs[api] = train_data_normalized[api][-train_window:].tolist()
print(test_inputs[5005072170100])

model.eval()
seq = {}

## оценка качества работы модели
for api in APIs:
    for i in range(fut_pred):
        seq[api] = torch.FloatTensor(test_inputs[api][-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs[api].append(model(seq[api]).item())

actual_predictions = {}
# преобразует данные обратно
for api in APIs:
    actual_predictions[api] = scaler[api].inverse_transform(np.array(test_inputs[api][train_window:] ).reshape(-1, 1))
print(actual_predictions[api])

x = np.arange(12, 24, 1)


for api in APIs:
    plt.title(f'API == {api}')
    plt.ylabel('Общее количество произведенной жидкой нефти')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(df[api][:12], label = 'Актуальные данные')
    plt.plot(x,actual_predictions[api], label = 'Предсказанные')
    plt.legend()
    # plt.show()
    plt.savefig(f'result/result_for_api_{api}')
    plt.clf()

