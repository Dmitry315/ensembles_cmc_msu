import os
from functools import wraps
import pickle
import pandas as pd
from flask import Flask, request, render_template, url_for, redirect, abort
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, FileField, FloatField, DecimalField, SubmitField, validators
from wtforms.fields.html5 import DecimalRangeField
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from ensembles import RandomForestMSE, GradientBoostingMSE

# from flask_sqlalchemy import SQLAlchemy
# from flask_restful import reqparse, abort, Api, Resource
# from flask_wtf import FlaskForm

# init app and configure server
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = '1515dd15dd3d5d1a51b5af515ca'

model_params = {}

UPLOAD_FOLDER = 'files/'
MODELS_FOLDER = 'models/'
app.config['MODEL_IDX'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
