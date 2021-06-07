#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:31:30 2021

@author: sacia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:06:37 2020
@author: randon
"""

#import plotly.express as px
#import plotly.graph_objs as go

from tensorflow import keras
import tensorflow as tf

from PIL import Image
import base64
from io import BytesIO

#####

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.backend import clear_session
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV

#import matplotlib.pyplot as plt
#import os
import numpy as np
import base64
import io
from io import BytesIO
import re
from dash import no_update

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.models import load_model



#####



import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app, server

import dash
import numpy as np

from app import app, server

from apps import home  #, page1


classes = ['cat', 'dog', 'rabbit']

enc = OneHotEncoder()
enc.fit([[0], [1], [2]]) 
def names(number):
    if(number == 0):
        return 'a cat'
    elif(number == 1):
        return 'a dog'
    elif(number == 2):
        return 'rabbit'




layout = html.Div([
    dbc.Container([
        
        dbc.Row([
            dbc.Col(html.H1("Welcome to Image Classification", className="text-center")
                    , className="mb-4 mt-4")
        ]),
        
             
        
        dbc.Row([
            dbc.Col(html.H4(children='Image Classification'
                                     ))
            ]),
        dbc.Row([
            
           dbc.Col(html.H5(children='You can click on the button to upload a new image:')                     
                    , className="mb-4")
            ]),
        
        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H5(children='Dog',
                                               className="text-center"),
                                       html.Img(src="/assets/dog.jpg", height="200px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Cat',
                                               className="text-center"),
                                       html.Img(src="./assets/cat.jpg", height="200px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Rabbit',
                                               className="text-center"),
                                       html.Img(src="/assets/rabbit.jpg", height="200px")])
                        ,className="mb-4"),
            ]),
        
        
        dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'background-color':'gray'
        },
        # Allow multiple files to be uploaded
        multiple=True
        ),
        
        #html.Div(id='output-image-upload', className="mb-4"),
        
        html.Div(id='output-image-upload', style={'textAlign': 'center'
        }),
    
        html.Div(id='prediction',style={'textAlign': 'center', 'font-size' : '60px'
        }),
        
        ])])

def parse_contents(contents):
    
    return html.Img(src=contents, style={'height':'200px', 'width':'200px'})


#@app.callback(Output('output-image-upload', 'children'),
             #Input('upload-image', 'contents'))

@app.callback([Output('output-image-upload', 'children'), Output('prediction', 'children')],
              [Input('upload-image', 'contents')])


def update_output(image):        
    
    if image is not None:
       
        children = parse_contents(image[0]) 
        img_data = image[0]
        img_data = re.sub('data:image/jpeg;base64,', '', img_data)
        img_data = base64.b64decode(img_data)  
        
        stream = io.BytesIO(img_data)
        img_pil = Image.open(stream)
        
        
        #Load model, change image to array and predict
        model = load_model('my_model.h5')
        dim = (150, 150)
        
        img = np.array(img_pil.resize(dim))
        
        x = img.reshape(1,150,150,3)

        answ = model.predict(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        
        pred = names(classification)   
        
                 
        return  children, pred
        
    else:
        return (no_update, no_update) 
    





        
        
   

