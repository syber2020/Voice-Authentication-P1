
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import os
# Create your views here.
from django.shortcuts import render
from django.views.generic import TemplateView
from django.conf import settings
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dropout, Activation
from sklearn.preprocessing import LabelEncoder
import IPython.display as ipd
# % pylab inline
import os
import pandas as pd
import librosa
import glob
import librosa.display
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import np_utils
from sklearn import metrics

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

import os


import tensorflow

model = open(os.path.join(settings.BASE_DIR, 'cnnspeakerRecog2_lab3.h5'))

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)

def RecordAudio(request):
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)

    X, sample_rate = librosa.load('output.wav',res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    l = []
    return render(request, 'record.html', context=None)



def extract_features1(file):
    # Sets the name to be the path to where the file is in my computer
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)

    # We add also the classes of each file as a label at the end

    return mfccs, chroma, mel, contrast, tonnetz

def PredictEngine(request):
    fs = 22050 # Sample rate
    seconds = 10  # Duration of recording
    print("Start recording")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.flac', fs, myrecording)
    features_label1 = extract_features1('output.flac')

    os.remove('output.flac')
    np.save('feature_labels1', features_label1)

    features_label1 = np.load('feature_labels1.npy', allow_pickle=True)

    os.remove('feature_labels1.npy')

    features1 = []
    features1.append(features_label1[0].tolist()+features_label1[1].tolist()+features_label1[2].tolist()+features_label1[3].tolist()+features_label1[4].tolist())

    # ss = StandardScaler()
    # X = ss.fit_transform(features1)
    opt = tensorflow.keras.optimizers.Adam(lr=0.00001, decay=1e-6)
    X = np.reshape(features1, (1, 193,1))
    # reconstructed_model = tensorflow.keras.models.load_model("cnnspeakerRecog.h5")
    json_file = open('modelcnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelcnn.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    preds = loaded_model.predict_classes(X)
    print("Predicted Speaker: ",preds)

    return render(request, 'predicted.html', {'preds':preds})
#def predIct(request)






# def getRecognized(audio):
#     import pickle
#     model = pickle.load(open("model.sav", "rb"))
#     scaled = pickle.load(open("scaler.sav", "rb"))

#     prediction = model.predict(sc.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))

#     if prediction == 0:
#         return "not survived"
#     elif prediction == 1:
#         return "survived"
#     else:
#         return "error"


# # our result page view
# def result(request):
#     pclass = int(request.GET['pclass'])
#     sex = int(request.GET['sex'])
#     age = int(request.GET['age'])
#     sibsp = int(request.GET['sibsp'])
#     parch = int(request.GET['parch'])
#     fare = int(request.GET['fare'])
#     embC = int(request.GET['embC'])
#     embQ = int(request.GET['embQ'])
#     embS = int(request.GET['embS'])

#     result = getPredictions(pclass, sex, age, sibsp, parch, fare, embC, embQ, embS)

#     return render(request, 'result.html', {'result':result})
