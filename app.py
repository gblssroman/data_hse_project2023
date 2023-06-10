# Интеграция со streamlit
import streamlit as st
bot_url = "https://t.me/official_studobot"
ipynb_url = "https://github.com/rtccreator/data_hse_project2023/blob/main/main.ipynb"
st.title("Итоговый проект / Наука о данных / Волобуев Владислав")
st.header("В чем заключается вкратце выполненная работа?")
st.subheader("\nНаписанный код может:\n\
1) Отправлять расписание на завтра и давать рекомендации,\n\
2) Узнавать актуальный курс юаня к рублю, отправлять график и рекомендовать, что делать для заработка,\n\
3) Принимать большие списки с контактными данными и отправлять список юзеру обратно только с валидными e-mail адресами и телефонами,\n\
4) Узнавать температуру на какую-либо дату в 2022 году в городе Базель основываясь на предсказаниях ML-модели,\n\
5) Создан телеграм-бот, который решает СЛАУ методом Крамера\n\n")

st.markdown(f'''<a href={bot_url}><button style="background-color:Red;color:White;border-radius:5px;padding:10px;margin:15px 0;border-color:Red;">Перейти в бота</button></a>''',
unsafe_allow_html=True)
st.header("\n\nВесь код:\n")
st.markdown(f'''<a href={ipynb_url}><button style="background-color:Green;color:White;border-radius:5px;padding:10px;margin:15px 0;border-color:Green;">Посмотреть исходный код ноутбука (Github)</button></a>''',
unsafe_allow_html=True)
#Далее код и его работа
####################################
import numpy as np
import sympy as sp
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from PIL import Image
from io import BytesIO
import io
import requests
import asyncio

bot_inst = Bot(st.secrets.tg.TG_TOKEN)
dp_inst = Dispatcher(bot_inst)

@dp_inst.message_handler(commands=['start'])
async def init_command(message: types.Message):
    reply_kb = InlineKeyboardMarkup()
    reply_kb.add(InlineKeyboardButton("🧮 Решить СЛАУ по Крамеру", callback_data='solve'))
    await message.answer("Добро пожаловать! Я телеграм-бот проекта Влада Волобуева, который умеет решать СЛАУ методом Крамера. Выберите действие:", reply_markup=reply_kb)

@dp_inst.callback_query_handler(lambda query: True)
async def process_query(callback_query: types.CallbackQuery):
    callback_id = callback_query.from_user.id
    callback_data = callback_query.data
    error_txt = "Ввод некорректный!"

    if callback_data == 'solve':
        await bot_inst.send_message(callback_id, "Введите СЛАУ в формате: [[2,5,-1,10], [1,-1,3,5], [3,2,4,4]]:")

@dp_inst.message_handler()
async def process_message(message: types.Message):
    message_text = message.text
    try:
        slau = eval(message_text)
        result = solve_cramer(slau)
        await message.answer(f"Результат: {result}")
    except Exception:
        await message.answer("Ввод некорректный!")

###функции

####основные функции бота
async def main():
    st.title("Бот, решающий СЛАУ методом Крамера успешно запущен!")
    st.write("По остальным пунктам - см. Jupyter Notebook (зеленая кнопка)! Там представлен весь код.")

async def start_polling():
    await dp_inst.start_polling()
####конец

def solve_cramer(slau):
    M = sp.Matrix(slau)
    M_coeff = M[:, :-1]
    M_const = M[:, -1]
    res = []
    
    if M_coeff.cols != M_coeff.rows:
        return 'Матрица не является квадратной, по Крамеру ее не решить, \
испольуйте, например, метод Гаусса'
    elif M_coeff.det() == 0:
        return 'Определитель основной матрицы равен 0, а это значит, \
что нет единственного решения!'
    else:
        for i in range(M_coeff.cols):
            M_temp = M_coeff.copy()
            M_temp[:, i] = M_const 
            #последовательно меняем столбцы матрицы на значения констант
            res.append(M_temp.det()/M_coeff.det())
            #добавляем к списку найденное значение переменной
            #после деления детерминанта на детерминант констант
            
    return res

###конец
    
if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(asyncio.gather(main(), start_polling()))


#END################################
