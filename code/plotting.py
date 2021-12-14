# -*- coding: utf-8 -*-
"""
Данный файл содержит график, который отражает все исходные данные (уже преобразованные)
все итоговые графики сохранены в api_img
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

PATH = os.path.join('../data', 'production.csv')
df = pd.read_csv(PATH)
df.head()

APIs = df['API'].unique()
APIs = np.random.choice(APIs, 10)

"""### Нарисуем график"""

for i in range(10):
    plt.figure(figsize=(13, 5))
    data = df[df['API'] == APIs[i]]

    production_ = 'b'
    raising = 'y'
    rolling_3 = 'r'
    rolling_5 = 'g'

    production = data['Liquid'] + data['Gas']
    x = np.array([int(i) for i in range(len(production))])

    ax = plt.bar(x, production, align='center')

    for j in range(1, len(ax)):
        if production.iloc[j - 1] * 1.1 < production.iloc[
            j]:  ## ставим повышение цвета на те, которые увеличились более чем на 10%
            ax[j].set_fc(raising)

            ## добавление легенд
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in [production_, raising, rolling_3, rolling_5]]
    labels = ["Производство", "Повышение", "Rolling Mean 3", 'Rolling Mean 5']

    plt.title(f'API = {APIs[i]}')
    plt.xlabel('Месяцы', fontsize=12)
    plt.ylabel('Производство', fontsize=12)
    plt.legend(handles, labels, prop={'size': 10})
    plt.grid(axis='y')

    ## добавим средства прокрутки с window = 3 и window = 5
    y = production.rolling(window=3, min_periods=1).mean()
    ax = plt.twinx()
    # ax.set_ylim(0, production.max())   
    ax.plot(x, y, color='r', label='Rolling Mean 3', linewidth=3)
    plt.gca().axison = False

    y = production.rolling(window=5, min_periods=1).mean()
    ax.plot(x, y, color='g', label='Rolling Mean 5')

    plt.savefig(f'api_img/data_result{APIs[i]}.png')
