import os
from functools import wraps
import pickle
import json
import pandas as pd
import plotly
import plotly.express as px
from flask import Flask, request, render_template, url_for, redirect, abort, send_file, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, FileField, FloatField, DecimalField, SubmitField, validators
from wtforms.fields.html5 import DecimalRangeField
from werkzeug.utils import secure_filename
from ensembles import RandomForestMSE, GradientBoostingMSE

# init app and configure server
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = '1515dd15dd3d5d1a51b5af515ca'

model_params = {}
files_to_delete = []

UPLOAD_FOLDER = 'files/'
MODELS_FOLDER = 'models/'
app.config['MODEL_IDX'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
