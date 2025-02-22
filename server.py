from flask import Flask
import logging
import socket
import os
from waitress import serve

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    serve(app, host='::', port=8080)
