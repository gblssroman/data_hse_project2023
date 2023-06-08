# Интеграция со streamlit
import streamlit as st
bot_url = "https://t.me/official_studobot"
st.title("Итоговый проект / Наука о данных / Волобуев Владислав")
st.header("В чем заключается проект?")
st.subheader("\nЭто универсальный бот-помощник для студента, который умеет:\n\
1) Отправлять расписание на завтра и давать рекомендации,\n\
2) Узнавать вероятность дождя на завтра и послезавтра,\n\
3) Узнавать актуальный курс юаня к рублю и отправлять график изменения,\n\
4) Переводить рукописный текст в напечатанный,\n\
5) Решать СЛАУ методом Крамера\n\n")

st.markdown(f'''<a href={bot_url}><button style="background-color:Red;color:White;border-radius:5px;padding:10px;margin:15px 0;border-color:Red;">Перейти в бота</button></a>''',
unsafe_allow_html=True)

st.header("\n\nВесь код:\n")
#Далее код и его работа

st.markdown('''# Итоговый проект / Наука о данных / Волобуев Владислав''')
st.markdown('''### В чем заключается проект: Это универсальный бот-помощник для студента, который умеет:<br>
1) Отправлять расписание на завтра и давать рекомендации,<br>
2) Узнавать актуальный курс юаня к рублю, отправлять график и рекомендовать, что делать для заработка,<br>
3) Принимать большие списки с контактными данными и отправлять список юзеру обратно только с валидными e-mail адресами и телефонами,<br>
4) Узнавать температуру на какую-либо дату в 2022 году в городе Базель основываясь на предсказаниях ML-модели,<br>
5) Решать СЛАУ методом Крамера<br>''')
st.markdown('''Импортируем необходимые нам библиотеки для общей работы:''')
import numpy as np
import sympy as sp
import locale
import random
import pandas as pd
import re
import csv
import openai
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from sklearn.linear_model import Ridge
import sqlite3
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ParseMode
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
st.markdown('''Это мы заполняем файл requirements.txt для того, чтобы Streamlit мог автоматом подгрузить нужные нам библиотеки для дальнейшего использования:''')
%pip freeze > requirements.txt
st.markdown('''### 1) Отправляем расписание на завтра и даем рекомендации, как лучше подготовиться к предметам
#### К сожалению, РУЗ ВШЭ закрыл доступ к API, поэтому расписание будем выдавать по Финансовому университету (API аналогичное один в один старому)

Используем оф. библиотеку openai и api-ключ, полученный из личного кабинета:''')
st.markdown('''#### Для начала передадим dummy-текст предметов в расписании, а после уже будем парсить API''')
subj_info = 'Математика, Английский язык, Программирование на C, Экономика'
def get_an_advice(prompt_subj):
    openai.api_key = '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'
    #важный момент - для работы данного кода необходим api, где лимит
    #не превышен. вполне возможно, что может быть достигнут максимум по этому ключу

    subj_tokens = int(len(prompt_subj.split(",")) * 50)

    #рассчитываем кол-во токенов по длине кол-ва предметов в какой-то день
    preprompt = f"Предложи максимально кратко, как \
    лучше подготовиться студенту к следующим дисциплинам: \
    В общей сумме должно получиться максимум {subj_tokens} символов! Очень кратко!"
    #st.write(preprompt)

    max_tokens = subj_tokens + 100
    #дадим модели больше, чем просим, чтобы не обрывать ответ

    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo', #выбираем самую быструю общедоступную модель 3.5 turbo
      messages=[
            {'role': 'system', 'content': preprompt},
            {'role': 'user', 'content': prompt_subj},
      ],
        max_tokens = max_tokens
    )
    
    return response['choices'][0]['message']['content']

response = get_an_advice(subj_info)
st.markdown('''#### Общая длина ответа в символах и сам ответ:''')
len(response)
st.write(response)
st.markdown('''### Теперь пропишем функции для парсинга: сначала мы ищем группу по запросу юзера, выдаем подсказки, а после отправляем расписание''')
def get_groups(query):
    url = f"https://ruz.fa.ru/api/search?term={query}&type=group"
    response = requests.get(url)
    grps_raw = response.json()
    #здесь генератором проходимся по json-ответу и берем значения id и label,
    #отсекая ненужное, в нашем случае ненужное - это ситуация, когда в label
    #перечислены сразу несколько групп, что нерелевантно
    res = [[grps_raw[i]['id'], 
            grps_raw[i]['label'].split(";")] for i in range(len(grps_raw)) if len(grps_raw) > 0 and len(grps_raw[i]['label'].split(";")) == 1]
    
    return res
def get_schedule(group_id, date):
    #обращаемся по id и дате, даты конца и начала равны
    url = f"https://ruz.fa.ru/api/schedule/group/{group_id}?start={date}&finish={date}&lng=1"
    response = requests.get(url)
    rasp_raw = response.json()
    #берем только название дисциплины, начало и конец по времени
    res = [[rasp_raw[i]['discipline'],
            rasp_raw[i]['beginLesson'],
            rasp_raw[i]['endLesson']] for i in range(len(rasp_raw)) if len(rasp_raw) > 0]
    
    res_final = []
    for i in res:
        if i not in res_final:
            res_final.append(i) #избавляемся от дубликатов
    return res_final
get_groups("пм20") #пример по запросу "пм20"
get_schedule(78637, "2023.06.06")
#обращаясь к чатгпт, выделим только названия предметов
tmrw_date = datetime.now() + timedelta(days=0)
tmrw_date = tmrw_date.strftime('%Y.%m.%d')
st.write(tmrw_date)
#получаем сегодняшнюю дату и прибавляем 1 день, используя timedelta из datetime
#после чего форматируем к нужному нам формату год.месяц.день

tmrw_schedule = get_schedule(78637, tmrw_date)
all_subj_prompt = ""

for i in tmrw_schedule:
    if i[0] not in all_subj_prompt:
        all_subj_prompt += f"{i[0]}, " #убираем повторения предметов и добавляем только их, без времени
all_subj_prompt
get_an_advice(all_subj_prompt) #видим советы по нашим дисциплинам, все работает!
st.markdown('''### 2) Парсинг технических прогнозов по курсу Юань/рубль (в дальнейшем по запросу пользователя через бота будут выдаваться актуальные)
#### Источники данных: Investing.com и TradingView

Используем Selenium:''')
st.markdown('''Используя клиент Хрома, подгружаем веб-драйвер Selenium и заставляем кликать на необходимые элементы для получения финального результата в виде списка, который будет затем "причесываться" и конвертироваться в читаемый формат.''')
driver = webdriver.Chrome()

driver.get('https://www.investing.com/currencies/cny-rub-technical')
time.sleep(2)
cookie_button = driver.find_elements(By.ID, 'onetrust-accept-btn-handler')
if len(cookie_button) > 0:
    cookie_button[0].click() #это мы нажимаем "продолжить" на куки-баннер, если он существует
data_time_list_inv = [300, 1800, 86400, 'week', 'month']
data_inv = []
for i in data_time_list_inv:
    time.sleep(1)
    #немного ждем и нажимаем на ссылки, которые находятся внутри li с атрибутом data-period
    driver.find_element(By.CSS_SELECTOR, f"li[data-period=\"{i}\"] > a").click()
    time.sleep(1)
    #искомое находится в div с этим id:
    target_div_inv = driver.find_element(By.ID, 'techStudiesInnerWrap')
    data_inv.append(list(map(lambda element: element.text, target_div_inv.find_elements(By.TAG_NAME, 'div'))))
    #парсим текст каждого внутреннего divа для 5 минут, получаса, дня, недели, месяца
st.write(data_inv)

driver.get('https://www.tradingview.com/symbols/CNYRUB/technicals/')
data_time_list_tdv = ['5m', '30m', '1D', '1W', '1M']
time.sleep(2)
data_tdv = []
for i in data_time_list_tdv:
    time.sleep(1)
    #аналогично с Инвестингом, только здесь ищем кнопку с определенным атрибутом
    #в id и после парсим с контейнера, добавляя в общий массив
    driver.find_element(By.CSS_SELECTOR, f"button[id=\"{i}\"]").click()
    time.sleep(1)
    data_tdv.append(driver.find_element(By.CLASS_NAME, 'countersWrapper-kg4MJrFB').text)
    #аналогично
st.write(data_tdv)

driver.quit()
st.write("Теперь адекватно скомпонуем полученное и сформируем из этого датафрейм")
#причесываем, убирая ненужные отступы
investing_data_clean = []
for i in data_inv:
    for j in i:
        j = j.replace("\n", "").split(":")
        investing_data_clean.append(j)
        
tradingview_data_clean = [s.split('\n') for s in data_tdv]
st.write(investing_data_clean, "\n\n", tradingview_data_clean)
st.markdown('''#### Манипуляциями в pandas мы получаем такое представление для показателей с Инвестинга:''')
time_periods = ['5 min', '30 min', 'day', 'week', 'month']
inv_data_pd = [investing_data_clean[i:i + 3] for i in range(0, len(investing_data_clean), 3)] 
inv_for_pd_final = list(map(list, zip(*inv_data_pd))) #поворачиваем получившийся список
df = pd.DataFrame(inv_for_pd_final, columns=time_periods)
st.dataframe(df = pd.DataFrame(inv_for_pd_final, columns=time_periods).split("=")[0].strip())
#преобразовываем в dataframe со столбцами наших временных интервалов
df
st.markdown('''#### Но нам будет проще преобразовать все в один словарь, т.к. структура данных разнится и мы так или иначе будем это отправлять юзеру в Telegram в текстовом формате
Объединим в один словарь:''')
final_yuan_dict = df.to_dict()

c = 0
for k, v in final_yuan_dict.items():
    final_yuan_dict[time_periods[c]][3] = tradingview_data_clean[c]
    c += 1
st.write(final_yuan_dict)
st.markdown('''#### Спарсим график юаня на момент запроса''')
driver = webdriver.Chrome()

driver.get('https://www.profinance.ru/charts/cnyrub/lc11')
time.sleep(2)
driver.find_element(By.ID, 'chart_button_plus').click() #немного увеличим масштаб
time.sleep(2)
img_elem = driver.find_element(By.ID, 'chart_img')
img_elem_src = img_elem.get_attribute('src')
response = requests.get(img_elem_src) #обрабатываем ссылку на изображение
img = Image.open(BytesIO(response.content)) #получаем наше изображение
#ниже - представление избражения в ноутбуке, в телеграме нам плотлиб здесь будет не нужен
plt.imshow(img)
plt.show()

driver.quit()
st.markdown('''### 3) Принимать большие списки с контактными данными и оставляем только с валидными e-mail адресами и телефонами (используем регулярные выражения).
#### CSV-файл нам помог сгенерировать: https://www.mockaroo.com/


В CSV-файле присутствуют как невалидные e-mailы, так и телефоны, есть еще и пустые клетки''')
df = pd.read_csv('dataset_emails_phones.csv')
st.dataframe(df = pd.read_csv('dataset_emails_phones.csv').split("=")[0].strip())
df
st.markdown('''#### Видим, что строк (людей) 100 + уже проглядываются заметные нарушения формата телефоннных номеров и имейлов''')
#обращаясь к регулярным выражениям, сделаем маски
phone_mask = re.compile(r'^\+7\d{10}$') #формат +7XXXXXXXXXX
email_mask = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$') #базовый формат имейла

def check_phone(phone):
    if phone_mask.match(phone):
        return True
    else:
        return False

def check_email(email):
    if email_mask.match(email):
        return True
    else:
        return False
#эти базовые функции мы применим на интересующие нас столбцы и сделаем два новых датафрейма
#типы всех значений нужно также рассмотреть как строки, иначе регулярные выражения не сработают
df['valid_phone'] = df['phone'].astype(str).apply(check_phone)
df['valid_email'] = df['email'].astype(str).apply(check_email)

valid_df = df[df['valid_phone'] & df['valid_email']] #объединяем строки, где нет ошибок

valid_df
st.markdown('''#### Теперь можем заметить, что строк осталось всего 71 и у всех из них все данные валидны, сконвертируем датафрейм в csv, т.к. нам еще отправлять это пользователю''')
valid_df.to_csv('valid_users.csv', index=False)
pd.read_csv('valid_users.csv') #все работает, csv создан корректно и без лишних индексов
st.markdown('''### 4) Анализируем датасет по погоде в городе Базель, Швейцария с 1 января 2010 года по 31 декабря 2022 года.
#### Источник данных: https://www.meteoblue.com/ru/%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D0%B0/archive/

Вот небольшая вырезка из датафрейма, полученного из CSV-файла:''')
df = pd.read_csv('all_weather_Bazel_ds.csv', skiprows=range(0, 9)) #строки 0-9 в датасете не несут никакой пользы для нас, 
st.dataframe(df = pd.read_csv('all_weather_Bazel_ds.csv', skiprows=range(0, 9)) #строки 0-9 в датасете не несут никакой пользы для нас, .split("=")[0].strip())
#поэтому мы их не учитываем
df.head()
st.write(len(df)) #текущий размер датафрейма
st.markdown('''#### Дадим столбцам другие названия. Заменим их соответственно по порядку на следующее для удобства.''')
new_weather_names = ["day", "temperature", "precipitation", "humidity", "wind", "cloud_cover", "sunshine_duration", "shortwave_radiation", "uv_radiation", "pressure"]
df.columns = new_weather_names
plt.figure(figsize=(8, 6))
x_new = np.linspace(0, 24, num=25) #зададим разделение по часам для графика
#выведем его красиво и информативно
plt.title('1 января 2010 года')
plt.xlabel('Время (ч)')
plt.ylabel('Температура (по Цельсию)')
plt.xticks(x_new, rotation=45, ha='right', fontsize=8)
plt.grid(True, linewidth=1)
st.pyplot(plt.plot(df['temperature'][0:25], linestyle='-', color='b'))
plt.show()
#вот как менялась температура 1 января 2010 года по часам
df['day'] = pd.to_datetime(df['day'], format='%Y%m%dT%H%M')
st.dataframe(df['day'] = pd.to_datetime(df['day'], format='%Y%m%dT%H%M').split("=")[0].strip())
# преобразовываем в формат datetime

df_1 = df.resample('D', on='day').mean().reset_index()
#находим среднее значение по каждому дню (имеем право, т.к. преобразовали
#к dataframe) и уберем старые индексы

df_1.head()
st.write(len(df_1)) #размер датафрейма сократился ровно в 24 раза
df_1.dtypes #все типы кроме дат преобразованы к флоату
#для предсказания погоды сделаем новый столбец target, в который для начала передадим температуру предыдущего дня
df_1['target'] = df_1.shift(-1)['temperature']
df_1 = df_1.ffill() #заполним NaN в последней клетке таргета предыдущим значением
df_1.iloc[:, 1:] = df_1.iloc[:, 1:].round(2) 
#и округляем все значения для адекватного представления, получаем:
weather = df_1
#weather.set_index('day', inplace=True)
weather
st.markdown('''#### Проверяем корреляцию показателей друг с другом''')
corr_weather = weather.corr()
corr_weather
st.markdown('''#### Построим неориентированный граф корреляций''')
correlation_matrix = corr_weather.corr().abs().round(2)
G = nx.from_pandas_adjacency(correlation_matrix)
pos = nx.spring_layout(G)

plt.figure(figsize=(12,12))

edge_colors = ["green" if G[u][v]['weight'] >= 0.6 and G[u][v]['weight'] < 1 else 'black' for u, v in G.edges()]
nx.draw(G, pos, with_labels=True, edge_color=edge_colors, font_weight='bold')

for i in pos:
    pos[i][1] += 0.02

edge_labels = nx.get_edge_attributes(G, "weight")
for (u, v), label in edge_labels.items():
    text_color = 'green' if label >= 0.6 and label < 1 else 'black'
    plt.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, label, 
             fontsize=10, color=text_color,
             horizontalalignment='center', verticalalignment='center')

plt.show()
#высокая корреляция (значения, совсем близкие к единице учитывать нет смысла)
#поэтому берем в таком рэнже
rows, cols = np.where((corr_weather > 0.6) & (corr_weather < 0.9))
for row, col in zip(rows, cols):
    st.write(corr_weather.index[row], "<->", corr_weather.columns[col], corr_weather.iloc[row, col])
st.markdown('''Показатели, которые коррелируют друг с другом больше всего, нас и интересуют''')
### FROM: https://github.com/dataquestio/project-walkthroughs/blob/master/temperature_prediction/predict.ipynb
rr = Ridge(alpha=.1)
predictors = weather.columns[~weather.columns.isin(["target"])]
#здесь нас не интересует только столбец "target", т.е. в предикторы
#пойдет все кроме даты и таргета

st.write(predictors)
#так и есть

def backtest(weather, model, predictors, start=730, step=20):
    #начинаем с 2012 года, пропускаем 730 дней, шаг предсказаний = 20
    #значения автора тем самым заменяем
    all_predictions = []
    
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors], train["target"])
        #тренируем все предикторы по таргету
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
st.dataframe(        preds = pd.Series(preds, index=test.index).split("=")[0].strip())
        combined = pd.concat([test["target"], preds], axis=1)
st.dataframe(        combined = pd.concat([test["target"], preds], axis=1).split("=")[0].strip())
        combined.columns = ["actual", "prediction"]
        #для наглядности здесь считаем разницу между фактом и предсказанием
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

predictions = backtest(weather, rr, predictors)
### END FROM
predictions
st.markdown('''#### Отправим результаты нашей работы в базу данных SQLite:''')
pred_2022 = predictions[predictions.index.year == 2022]
with sqlite3.connect('db_ml_res.db') as conn:
    pred_2022.to_sql('weather_data', conn, if_exists='replace', index=True, index_label='date')
st.markdown('''#### Теперь создадим функцию, где выведем массив, где показано, какие были предсказание и погода на этот день в Базеле:''')
def forecast_weather(date):
    if len(date.split(" ")) == 2:
        date += " 2022"
        #добавляем 2022, т.к. предсказываем именно по этому году
        #и конвертируем в datetime
        date = pd.to_datetime(date, format="%d %B %Y", dayfirst=True)
st.dataframe(        date = pd.to_datetime(date, format="%d %B %Y", dayfirst=True).split("=")[0].strip())
        date_sql = date.strftime('%Y-%m-%d')
        
        with sqlite3.connect('db_ml_res.db') as sql_conn:
            st.write(date_sql)
            res = pd.read_sql_query(f"SELECT * FROM weather_data WHERE date = '{date_sql} 00:00:00'", sql_conn)
st.dataframe(            res = pd.read_sql_query(f"SELECT * FROM weather_data WHERE date = '{date_sql} 00:00:00'", sql_conn).split("=")[0].strip())
        return [date_sql, f"{round(res['prediction'][0], 1)} C", f"{round(res['actual'][0], 1)} C"]
    else:
        return "Введите действительную дату в формате: день месяц"

forecast_weather("12 Июнь")
st.markdown('''### 5) Решаем СЛАУ методом Крамера
#### Используем sympy, на вход принимаем коэффициенты и константы, а если матрица не будет квадратной или определитель системы будет равен 0, скажем, что нужно использовать метод Гаусса, а по Крамеру здесь решить не получится


Возьмем произвольную систему для примера и составим функцию вида:''')
st.markdown('''2x + 5y - z = 10,<br>
x - y + 3z = 5,<br>
3x + 2y + 4z = 4<br>
Можем представить в виде расширенной матрицы (последний столбец - коэффициенты);''')
M1 = sp.Matrix([[2,5,-1,10],
               [1,-1,3,5],
               [3,2,4,4]])
M1
st.markdown('''Т.к. мы будем это все интегрировать в бота, сделаем проще: будем принимать список вида: [[2,5,-1,10],
               [1,-1,3,5],
               [3,2,4,4]]. <br>Тогда:''')
M_1 = [[2,5,-1,10], [1,-1,3,5], [3,2,4,4]]

def solve_slau_by_cramer(slau):
    M = sp.Matrix(slau)
    M_coeff = M[:, :-1]
    M_const = M[:, -1]
    res = []
    
    if M_coeff.cols != M_coeff.rows:
        return 'Матрица не является квадратной, по Крамеру ее не решить, \
испольуйте, например, метод Гаусса'
    elif M_coeff.det() == 0:
        return 'Определитель основной матрицы равен 0, а это значит, \
что нет единственного решения!''
    else:
        for i in range(M_coeff.cols):
            M_temp = M_coeff.copy()
            M_temp[:, i] = M_const 
            #последовательно меняем столбцы матрицы на значения констант
            res.append(M_temp.det()/M_coeff.det())
            #добавляем к списку найденное значение переменной
            #после деления детерминанта на детерминант констант
            
    return res

solve_slau_by_cramer(M_1) #видим, что единственного решения нет
M_2 = [[2,3,-1,5], [4,-1,2,1], [-1,1,3,4]]
solve_slau_by_cramer(M_2) #а здесь мы получаем решение в виде списка переменных


from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import asyncio
from aiogram.types import *

bot = Bot(st.secrets.tg.TG_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply("Привет, студент! Добро пожаловать! Вот мое меню:\n"
                        "1) Узнать расписание на завтра\n"
                        "2) Узнать вероятность дождя на завтра-послезавтра\n"
                        "3) Узнать курс юаня и увидеть график\n"
                        "4) Перевести рукописный текст в напечатанный\n"
                        "5) Решить СЛАУ методом Крамера")
    keyboard = InlineKeyboardMarkup(row_width=2)
    buttons = [
        InlineKeyboardButton("Расписание", callback_data="schedule"),
        InlineKeyboardButton("Погода", callback_data="weather"),
        InlineKeyboardButton("Курс юаня", callback_data="currency"),
        InlineKeyboardButton("Перевод текста", callback_data="translate"),
        InlineKeyboardButton("Решение СЛАУ", callback_data="solve")
    ]
    keyboard.add(*buttons)
    await message.answer("Выберите действие:", reply_markup=keyboard)


@dp.callback_query_handler(lambda c: c.data == 'schedule')
async def schedule(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, "Введите номер группы:")

@dp.callback_query_handler(lambda c: c.data == 'weather')
async def weather(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    rain_probability = await get_rain_probability()
    await bot.send_message(callback_query.from_user.id, f"Вероятность дождя на завтра: {rain_probability}%")

@dp.callback_query_handler(lambda c: c.data == 'currency')
async def currency(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    yuan_rate, graph_path = await get_yuan_rate_and_graph()
    with open(graph_path, 'rb') as photo:
        await bot.send_photo(callback_query.from_user.id, photo, caption=f"Курс юаня: {yuan_rate}")

@dp.callback_query_handler(lambda c: c.data == 'translate')
async def translate(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, "Отправьте фото рукописного текста:")

@dp.callback_query_handler(lambda c: c.data == 'solve')
async def solve(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, "Введите матрицу СЛАУ:")

async def main():
    st.title("STODOBOT Telegram - launching...")
    st.write("Добро пожаловать!")

async def start_polling():
    await dp.start_polling()

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(asyncio.gather(main(), start_polling()))

