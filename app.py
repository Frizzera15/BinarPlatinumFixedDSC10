'''
import re
import pickle
import pandas as pd
import numpy as np
import sqlite3
import nltk 

from flask import Flask, jsonify, request, render_template, redirect, url_for

from data_cleansing import text_normalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, save_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from flasgger import Swagger
from flasgger import swag_from


TABLE_NAME = "tweet_cleaning"

# create flask object
app = Flask(__name__, template_folder='templates')
cv=CountVectorizer()
le=LabelEncoder()


def loading_all_files():
    tokenizer = pickle.load(open('data/tokenizer.pkl','rb'))
    nlp_objects = pickle.load(open('data/mlp_classifier.pkl','rb'))
    #model_cnn = pickle.load(open('data/model.h5','rb'))
    #model_lstm = pickle.load(open('data/model_lstm.h5','rb'))
    

    return tokenizer, nlp_objects, #model_cnn, model_lstm

tokenizer, nlp_objects = loading_all_files()

swagger_config = {
    "headers": [],
    "specs": [{"endpoint":"docs", "route": '/docs.json'}],
    "static_url_path": "/flasgger_static",
    "swagger_ui":True,
    "specs_route":"/docs/"
}

swagger = Swagger(app,
                  config = swagger_config
                 )


def predict_paragraph(model, model_no, paragraph, cv, le):

    if model_no in [1, 2]:
        paragraph = text_normalization(paragraph)
        test_data_transformed = cv.transform([paragraph])
        y_pred = model.predict(test_data_transformed)
        y_pred = le.inverse_transform(y_pred)
        probability = 0
        return y_pred[0], probability

    #elif model_no in [3,4]:
        paragraph = text_normalization(paragraph)
        paragraph = tokenizer.texts_to_sequences([paragraph])
        padded_paragraph = pad_sequences(paragraph,padding='post',maxlen=input_len)
        
        y_pred = model.predict(padded_paragraph, batch_size=1)

        probability = np.max(y_pred, axis=1)

        y_pred = onehot.inverse_transform(y_pred).reshape(-1)
        
        
        return y_pred[0], probability[0]
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        input_text = request.form['text_input']
        option1 = request.form.get('option1')
        option2 = request.form.get('option2')
        option3 = request.form.get('option3')
        option4 = request.form.get('option4')
        
        model = None
        model_no = None
        
        if option1 is not None:
            model = tokenizer
            model_no = 1
        elif option2 is not None:
            model = nlp_objects
            model_no = 2
        elif option3 is not None:
            model = model_cnn
            model_no = 3
        elif option4 is not None:
            model = model_4
            model_no = 4

        # Call predict_paragraph and pass cv and le as arguments
        hasil_prediction, probability = predict_paragraph(model=model,
                                                         model_no=model_no,
                                                         paragraph=input_text,
                                                         cv=cv,  # Pass the CountVectorizer
                                                         le=le   # Pass the LabelEncoder
                                                         )

        json_response = {'paragraph': input_text,
                         'predicted_category': hasil_prediction,
                         'probability': f'{int(probability * 100)}%'
                         }

        json_response = jsonify(json_response)
        return json_response
    else:
        return render_template("homepage.html")


#@app.route('/', methods=['GET', "POST"])
#def hello_world():
    #if request.method == 'POST':
        input_text = request.form['text_input']
        option1 = request.form.get('option1')
        option2 = request.form.get('option2')
        option3 = request.form.get('option3')
        option4 = request.form.get('option4')
        
        model = None
        model_no = None
        
        if option1 is not None:
            model = tokenizer
            model_no = 1
        elif option2 is not None:
            model = nlp_objects
            model_no = 2
        elif option3 is not None:
            model = model_cnn
            model_no = 3
        elif option4 is not None:
            model = model_4
            model_no = 4

        hasil_prediction, probability = predict_paragraph(model = model,
                                             model_no = model_no,
                                             paragraph = input_text
                                            )

        json_response={'paragraph': input_text,
                       'predicted_category': hasil_prediction,
                       'probability': f'{int(probability*100)}%'
                      }

        json_response=jsonify(json_response)
        return json_response
    #else:
        return render_template("homepage.html")

if __name__ == '__main__':
    app.run(debug=True)
'''
import re
import pickle
import pandas as pd
import numpy as np
import sqlite3
import nltk 

from flask import Flask, jsonify, request, render_template, redirect, url_for
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from PlatinumGroup1BaseData import textcleansing
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, save_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from flasgger import Swagger
from flasgger import swag_from

TABLE_NAME = "tweet_cleaning"

# Create a Flask object
app = Flask(__name__, template_folder='templates')
#cv = CountVectorizer()
#le = LabelEncoder()

def loading_all_files():
    tokenizer = pickle.load(open('data/tokenizer.pkl', 'rb'))
    nlp_objects = pickle.load(open('data/model_mlpc.h5', 'rb'))
    cv = pickle.load(open('data/CountVectorizer.pkl', 'rb'))
    le = pickle.load(open('data/LabelEncoder.pkl', 'rb'))
    onehot = pickle.load(open('data/onehot.pkl','rb'))
    input_len = pickle.load(open('data/input_len.pkl','rb'))
    #pad_sequence = pickle.load(open('data/pad_sequence.pkl', 'rb'))
    model_lstm = load_model('data/lstmmodels.h5')
    #model_lstm = pickle.load(open('data/lstmmodels.h5','rb'))
    
    return tokenizer, nlp_objects, cv, le, onehot, model_lstm, input_len

tokenizer, nlp_objects, cv, le, onehot, model_lstm, input_len = loading_all_files()

swagger_config = {
    "headers": [],
    "specs": [{"endpoint": "docs", "route": '/docs.json'}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, config=swagger_config)

def predict_paragraph(model, model_no, paragraph):
    if model_no in [1, 2]:
        paragraph = text_normalization(paragraph)
        test_data_transformed = cv.transform([paragraph])
        y_pred = model.predict(test_data_transformed)
        y_preds = le.inverse_transform(y_pred)
        probability = model.predict_proba(test_data_transformed)
        return y_preds[0], probability [0] [1]
    
    elif model_no in [3,4]:
        paragraph1 = text_normalization(paragraph)
        paragraph2 = tokenizer.texts_to_sequences([paragraph1])
        print (paragraph2)
        padded_paragraph = pad_sequences(paragraph2,padding='post',maxlen=input_len)
        
        y_pred = model.predict(padded_paragraph, batch_size=1)

        probability = np.max(y_pred, axis=1)

        y_pred = onehot.inverse_transform(y_pred).reshape(-1)
        return y_pred[0], probability[0]
    else:
        return "Model not supported", 0

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        input_text = request.form['text_input']
        model_option = request.form.get('model_option')
        
        model = None
        model_no = None
        
        if model_option == 'option1':
            model = nlp_objects
            model_no = 1
        elif model_option == 'option2':
            model = model_lstm
            model_no = 3

        hasil_prediction, probability = predict_paragraph(model=model,
                                                         model_no=model_no,
                                                         paragraph=input_text)
        
        json_response = {'paragraph': input_text,
                         'predicted_category': hasil_prediction,
                         'probability': f'{int(probability * 100)}%'
                         }

        json_response = jsonify(json_response)
        return json_response
    else:
        return render_template("homepage.html")

if __name__ == '__main__':
    app.run(debug=True)
