import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import string
from PIL import Image
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
import re
from glob import glob
import io
import cv2
from flask_mysqldb import MySQL
from flask_dropzone import Dropzone
import pytesseract
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask import Flask, render_template, request, session, jsonify, send_file, flash, url_for, redirect
import requests
import MySQLdb
import sys
import matplotlib.pyplot as plt
import tensorflow as tf


basedir = os.path.abspath(os.path.dirname("C:\\Users\\Rishwa\\Desktop\\image_caption\\"))
global f
app = Flask(__name__, template_folder='Template/')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config["MONGO_URI"]="mongodb://localhost:27017/caption"
mongodb_client=PyMongo(app)
db = mongodb_client.db

app.config['SECRET_KEY'] = 'itsimagecaption'

bcrypt=Bcrypt(app)
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static'),
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=9,
    DROPZONE_MAX_FILES=30,
    DROPZONE_REDIRECT_VIEW='completed'  # set redirect view
)

dropzone = Dropzone(app)


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

filename = "C:\\Users\\Rishwa\\Desktop\\image_captioning\\Flickr8k_text\\Flickr8k.token.txt"
# load descriptions
doc = load_doc(filename)
print(doc[:300])

def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Original Vocabulary Size: %d' % len(vocabulary))

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_descriptions(descriptions, 'descriptions.txt')

def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load training dataset (6K)
filename = 'C:\\Users\\Rishwa\\Desktop\\image_captioning\\Flickr8k_text\\Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# Load the inception v3 model
model = InceptionV3(weights='imagenet')

# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

# Create a list of all the training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1 # one for appended 0's

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = list(), list(), list()
                n=0

# Load Glove vectors
glove_dir = 'C:\\Users\\Rishwa\\Desktop\\image_captioning\\@gdriveit_bot.glove.6B'
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights('C:\\Users\\Rishwa\\Desktop\\image_captioning\\model.h5')

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

@app.route('/', methods=['GET', 'POST'])
def index1():
    return render_template("register.html")

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('imageupload.html')


@app.route('/completed', methods=['GET', 'POST'])
def completed():
    # full_filename = os.path.join(app.config['UPLOADED_PATH'], f.filename)
    print("here")
    db.user.update_one({"name":sessionname} ,
                       { '$push' : {"image": {"name": f.filename , "caption" : "caption"}  }})
    # print(full_filename)
    model = load_model('abc.h5')
    return render_template("display.html", user_image="done")
    # return render_template('display.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    print("in upload")
    if request.method == 'POST':
        print("in post")

        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
        db.image.insert_one({"username": account['_id'], "image": f.filename })
    return render_template('imageupload.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'uname' in request.form and 'pwd' in request.form and 'emailid' in request.form and 'pwd' in request.form and 'mno' in request.form:
        # Create variables for easy access
        username = request.form['uname']
        password = request.form['pwd']
        email = request.form['emailid']
        pwd = request.form['pwd']
        pwd1 = bcrypt.generate_password_hash(pwd)
        cpwd = request.form['cpwd']
        mno = request.form['mno']

        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        elif cpwd != pwd:
            msg = 'please enter same password'
        else:
            db.user.insert_one({"name":username , "pwd": pwd1  , "email" : email , "mno":mno})
        return render_template("register.html")
    return render_template("register.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    mssg = ""
    if request.method == 'POST' and 'uname' in request.form and 'pwd' in request.form:
        # Create variables for easy access
        username = request.form['uname']
        password = request.form['pwd']

        global account
        account = db.user.find_one({"name":username})
        print(account)
        # If account exists in accounts table in out database
        if account and bcrypt.check_password_hash(account['pwd'],password) :
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['uname'] = account['name']
            global sessionname
            sessionname = session['uname']
            print(account['_id'])
            # session['_id']=account['_id']
            return render_template('home.html')
        else:
            # Account doesnt exist or username/password incorrect
            mssg = "*Incorrect username/password"
            return render_template('login.html', msg=mssg)
    return render_template('login.html', msg=mssg)

if __name__ == "__main__":
    app.run(debug=True)