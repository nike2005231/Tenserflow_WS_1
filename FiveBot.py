import telebot 
from PIL import Image
from telebot import types
import io
import os
from random import *
import time
import re
from currency_converter import CurrencyConverter

token = "6370383115:AAEfZxMCflGS-cCUQ-hiC_LgdK8NU_QnZ9A"

bot = telebot.TeleBot(token)

currency = CurrencyConverter()

con = 0 # Ограничитель для смс в all обработчике

#button
markup = types.InlineKeyboardMarkup()
markup_rep = types.ReplyKeyboardMarkup()

Button_photo_bw = types.InlineKeyboardButton("Перевод фото в чб", callback_data='bw')
Button_calc = types.InlineKeyboardButton("Калькулятор", callback_data='calc')
Button_curse = types.InlineKeyboardButton("Конвертер валют", callback_data='curse')

Button_start = types.KeyboardButton("/start")

def un_pack_button():
    global markup
    global markup_rep
    
    markup_rep = types.ReplyKeyboardMarkup()
    markup_rep.add(Button_start)

    markup = types.InlineKeyboardMarkup()
    markup.row(Button_curse)
    markup.row(Button_photo_bw, Button_calc)
    

#ФУНКЦИЯ ДЛЯ КНОПОК
def photo_id(message):
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        file = bot.download_file(file_path)

        photo_ids = {}

        photo_ids[file_path] = message.chat.id

        image = Image.open(io.BytesIO(file))

        bw_image = image.convert('L')

        temp_path = 'temp.jpg'
        bw_image.save(temp_path)

        bot.send_message(message.chat.id, f"Ожидайте")

        if file_path in photo_ids and message.chat.id == photo_ids[file_path]:
            with open(temp_path, 'rb') as photo:
                bot.send_photo(photo_ids[file_path], photo)
            os.remove(temp_path)
            bot.send_message(message.chat.id, "Фото обработано")
            
        else:
            bot.send_message(message.chat.id, "Произошла ошибка при обработке фото")
    except:
        bot.send_message(message.chat.id, "Нужно было отправить фотографию")


def but_calc(call):
    bot.send_message(call.message.chat.id, "Введите математическое выражение для вычисления (например, 2+2):")
    bot.register_next_step_handler(call.message, calculate)

def calculate(message):
    allowed_chars = set("0123456789+-*/.() ")

    try:
        expression = message.text

        if not set(expression).issubset(allowed_chars):
            raise ValueError("Введены недопустимые символы.")

        dangerous_operators = set(";%{}[]")
        if any(op in expression for op in dangerous_operators):
            raise ValueError("Введены недопустимые операторы.")

        result = eval(expression)
        bot.send_message(message.chat.id, f"Результат: {result}")
    except ValueError as e:
        bot.send_message(message.chat.id, f"Произошла ошибка при вычислении. {str(e)}")
    except Exception as e:
        bot.send_message(message.chat.id, "Произошла неожиданная ошибка при вычислении.")


@bot.message_handler(commands=["start"])
def Start(mes):
    un_pack_button()
    bot.send_message(mes.chat.id, f"Привет, {mes.from_user.first_name}!", reply_markup=markup_rep)
    bot.send_message(mes.chat.id, "<b> Выбери варианты действий: </b> ", reply_markup=markup, parse_mode="html")


@bot.callback_query_handler(func=lambda call: True)
def gen_but(call):
    if call.data == 'bw':
        bot.send_message(call.message.chat.id, "Отправьте фото(Не в виде файла)")
        bot.register_next_step_handler(call.message, photo_id)

    elif call.data == 'calc':
        but_calc(call)
        
    elif call.data == "curse":
        bot.send_message(call.message.chat.id, "Чтобы перевести деньги из одной валюты в другую, напишите мне сообщение вот в таком формате: Перевести 100 usd eur")

@bot.message_handler()
def all(mes):
    global con
    
    list_ran = ["Допустим", "Угу", "Возможно", "Ничего не скажу по этому поводу", "Бывает", "хех", "Да", "...", "ууууу", "Обработал", "нет"]
    list_mat = ["пидр", "хуй", "сук", "еблан", "долбаеб", "хуе", "пидор", "еба", "пизд", "уеб", "бля"]

    dick_big = {
                "привет": "Привет", "пока": "Пока", "хоч":"Хоти", "дела": "Нормально, не хорошо, не плохо... Нормально",
                "врешь": "Нет конечно", "как дела?": "Хорошо, спасибо!", "что делаешь": "Отвечаю на сообщения", "ты кто?": "Я бот написаный одним гением",
                "как тебя зовут": "Как удобно так и называй", "что нового": "Ничего особенного", "ты умный": "Я стараюсь быть умным",
                "что ты умеешь": "Я могу отвечать на сообщения и выполнять различные задачи", "время": f"Сейчас : {time.ctime()}",
                "любишь": "Люблю", "ку": "Привет", "здорова": "Здоровее видали", "салам": "Ну привет", "време": f"Сейчас : {time.ctime()}", "час": f"Сейчас : {time.ctime()}"
                }
    
    con = 0
    
    mes_pepl = mes.text.lower()
    mes_pepl = str(mes_pepl)
    
    try:
        if mes_pepl.find("перевести") != -1: #перевести 100 RUB USD
            x = mes_pepl.split()
            res = currency.convert(x[1], x[2].upper(), x[3].upper())
            bot.send_message(mes.chat.id, f"{x[1]} {x[2].upper()} в {x[3].upper()} = {round(res, 2)}")
            con += 1
    except:
        bot.send_message(mes.chat.id, "Не вышло попробуйте еще раз написать запрос в формате: Перевести 100 RUB USD")
        bot.send_message(mes.chat.id, "Возможная проблема в курсе рубля так-как он может быть не обновлен")
        con += 1
    
    for key, value in dick_big.items():
        if mes_pepl.find(key) != -1:
            bot.reply_to(mes, f"{value}, {mes.from_user.first_name}!")
            con += 1
            break
        
    for x in list_mat:
        if mes_pepl.find(x) != -1:
            bot.send_message(mes.chat.id, f"Материться плохо, {mes.from_user.first_name}!")
            con += 1
            break
    if con < 1:    
        bot.send_message(mes.chat.id, f"{choice(list_ran)}")
        con = 0   
        

bot.polling(non_stop=True)