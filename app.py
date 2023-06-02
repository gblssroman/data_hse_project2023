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

print(TG_TOKEN)
