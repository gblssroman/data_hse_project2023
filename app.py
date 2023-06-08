# Интеграция со streamlit
import streamlit as st
bot_url = "https://t.me/official_studobot"
ipynb_url = "https://github.com/rtccreator/data_hse_project2023/blob/main/main.ipynb"
st.title("Итоговый проект / Наука о данных / Волобуев Владислав")
st.header("В чем заключается проект?")
st.subheader("\nЭто универсальный бот-помощник для студента, который умеет:\n\
1) Отправлять расписание на завтра и давать рекомендации,\n\
2) Узнавать актуальный курс юаня к рублю, отправлять график и рекомендовать, что делать для заработка,\n\
3) Принимать большие списки с контактными данными и отправлять список юзеру обратно только с валидными e-mail адресами и телефонами,\n\
4) Узнавать температуру на какую-либо дату в 2022 году в городе Базель основываясь на предсказаниях ML-модели,\n\
5) Решать СЛАУ методом Крамера\n\n")

st.markdown(f'''<a href={bot_url}><button style="background-color:Red;color:White;border-radius:5px;padding:10px;margin:15px 0;border-color:Red;">Перейти в бота</button></a>''',
unsafe_allow_html=True)
st.markdown(f'''<a href={ipynb_url}><button style="background-color:Red;color:White;border-radius:5px;padding:10px;margin:15px 0;border-color:Green;">Посмотреть исходный код ноутбука (Github)</button></a>''',
unsafe_allow_html=True)

st.header("\n\nВесь код:\n")
#Далее код и его работа

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
async def yuan_ruble(callback_query: types.CallbackQuery):
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

