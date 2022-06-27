from flask import Flask, render_template
import sys
import os
import sqlite3
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.model import model as re

conn = sqlite3.connect('flask_app/model.db')
cur = conn.cursor()
df = pd.read_sql_query("SELECT * FROM steam_data;",conn)

app = Flask(__name__)

@app.route('/')
def index():


    return render_template('home.html')

@app.route('/<num>')
def search_id(num):
    x = re(df,num)
    title_list = []
    for x in x["Title"]:
        title_list.append(x)
    
    return render_template('index.html',title_list=title_list)


