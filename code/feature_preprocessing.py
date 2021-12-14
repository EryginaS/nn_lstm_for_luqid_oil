# -*- coding: utf-8 -*-
"""
Курсовая работа Ерыгина С.Н АСМ 21-04
В данном файле описаны все преобразования данных, файлы с интерпретацией данных находится в data.png
Корреляция изображена на correlation.png

"""
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

WELLS_DIR_PATH = os.path.join('data', 'wells_info.csv')
wells = pd.read_csv(WELLS_DIR_PATH)
print('wells', wells)

PRODUCTION_DIR_PATH = os.path.join('data', 'production.csv')
production = pd.read_csv(PRODUCTION_DIR_PATH)
print('productions', production)

"""
### Создадим колонки с общим производством жидкости за все время и за первые 12 месяцев
"""

APIs = production['API'].unique()
total_all_time = {}  ## вся продукция за все время
for api in APIs:
    total_all_time[production[production["API"] == api]["Liquid"].sum()] = api

APIs = production['API'].unique()
total_first_year = {}  ## объем производства за первый год
for api in APIs:
    total_first_year[production[production["API"] == api]["Liquid"].iloc[:12].sum()] = api

wells['total_all_time'] = total_all_time
wells['total_first_year'] = total_first_year

# делит данные на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(wells.iloc[:, :-2], wells['total_first_year'],
                                                    test_size=0.3, random_state=11)
X_train.to_csv(os.path.join('data', 'wells_info_train.csv'), header=False)
X_test.to_csv(os.path.join('data', 'wells_info_test.csv'), header=False)

PATH = os.path.join('data',
                    'wells_info_2.0.csv')  ## новый файл csv with total_all_time and total_first_year features
wells.to_csv(PATH)

"""### Извлечение признаков"""

sns.pairplot(wells[wells.columns[:]])
# plt.savefig('data.png')
"""###### Мы четко видим некоторые корреляции"""

corrs = wells.corr()
mask = np.zeros_like(corrs, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corrs, mask=mask, vmax=1, center=0, annot=True, fmt='.3f',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
# plt.savefig('correlation.png')
plt.show()
"""
Мы видим высокие корреляции в in LatWGS84 - BottomHoleLatitude, LonWGS84 - BottomHoleLogtude, LatWGS84 - PROP_PER_FOOT, LatWGS84 - WATER_PER_FOOT, BottomHoleLatitude - PROP_PER_FOOT, BottomHoleLatitude - WATER_PER_FOOT, PROP_PER_FOOT - WATER_PER_FOOT 
Не обращайте внимания на корреляцию total_all_time - total_first_year, поскольку они оба рассчитываются другими функциями, а total_all_time включает total_first_year
"""

print(wells.head())

X_train = pd.DataFrame.sparse.from_spmatrix(X_train)
print('x_train shape', X_train.shape)

X_test = pd.DataFrame.sparse.from_spmatrix(X_test)
print('X_test.shape', X_test.shape)
