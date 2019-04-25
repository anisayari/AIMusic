import pandas as pd
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
#Keras
import keras
import numpy as np
from keras import models
from keras import layers
import librosa
import os

def extract_features_from_a_song(x,sr,name_of_song):
    dict_features = {}
    #---------------spectral_centroid
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
    dict_features['spec_cent'] = np.mean(spectral_centroids)
    #spectral_centroids.shape
    # Computing the time variable for visualization
    # frames = range(len(spectral_centroids))
    #t = librosa.frames_to_time(frames)
    # Plotting the Spectral Centroid along the waveform
    #librosa.display.waveplot(x, sr=sr, alpha=0.4)
    #plt.plot(t, normalize(spectral_centroids), color='r')
    #plt.savefig('data/images/spectral_centroid/spectral_centroid_{}.png'.format(name_of_song))
    #plt.clf()
    #---------------spectral_roll_off
    spectral_rolloff = librosa.feature.spectral_rolloff(x,sr=sr)
    dict_features['spec_rolloff'] = np.mean(spectral_rolloff)

    #librosa.display.waveplot(x, sr=sr, alpha=0.4)
    # plt.plot(t, normalize(spectral_rolloff), color='r')
    #plt.savefig('data/images/spectral_centroid/spectral_rolloff_{}.png'.format(name_of_song))
    #plt.clf()
    # ---------------spectral_bandwith
    spectral_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
    dict_features['spec_bw'] = np.mean(spectral_bw)
    # ---------------zero_cross_rating
    zcr = librosa.feature.zero_crossing_rate(x)
    dict_features['zcr'] = np.mean(zcr)
    #---------------MFCC
    mfcc = librosa.feature.mfcc(x, sr=sr)
    dict_features['mfcc'] = np.mean(mfcc)
    # Displaying  the MFCCs:

    #librosa.display.specshow(sklearn.preprocessing.scale(mfcc, axis=1), sr=sr, x_axis='time')
    #plt.savefig('data/images/mfcc/mfcc_scaled_{}.png'.format(name_of_song))
    #plt.clf()
    #---------------chroma_frequencies
    chroma_stft = librosa.feature.chroma_stft(x, sr=sr)
    dict_features['chroma_stft'] = np.mean(chroma_stft)
    #plt.figure(figsize=(15, 5))
    #librosa.display.specshow(chroma_stft , x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    #plt.savefig('data/images/chroma_frequencies/chroma_frequencies_{}.png'.format(name_of_song))
    #plt.clf()

    return dict_features

def run_on_oldtow():
    dict_features = {}
    dict_features['chroma_stft'] = 0
    dict_features['mfcc'] = 0
    dict_features['spec_cent'] = 0
    dict_features['spec_rolloff'] = 0
    dict_features['spec_bw'] = 0
    dict_features['zcr'] = 0
    audio_path = 'Old_Town_Road.mp3'
    exists = os.path.isfile(audio_path)
    if exists :
        x, sr = librosa.load(audio_path)
        uuid = 'oldtownroad'
        try:
            pass
            #create_waveform_image(x, sr, uuid)
        except OverflowError:
            print('Cannot save waveform file for',uuid)
        try:
            pass
            #create_spectrogram_image(x,sr,uuid)
        except OverflowError:
            print('Cannot save spectogram file for',uuid)
        try:
            dict_features = extract_features_from_a_song(x,sr, uuid)
        except OverflowError:
            print('Cannot run features extract for',uuid)
        print('{} DONE'.format(uuid))
    print(dict_features)
    oldtown = pd.DataFrame(dict_features, index=[0])
    oldtown.to_csv('oldtown.csv',header=True,index=False,sep=";")
    return oldtown

#run_on_oldtow()

data = pd.read_csv('data/tracks_list.csv',header=0,sep=";")
data.head()
# Dropping unneccesary columns
data = data.drop(['uuid','artist','title_music','lastfm_music_url','lastfm_artist_url','videoID_youtube'],axis=1)
data = data[data['spec_cent']!=0]
genre_list = data['style']
data = data.drop("style",axis=1)
print(data.columns)
encoder = LabelEncoder()

y = encoder.fit_transform(genre_list)
print(y)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data, dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,epochs=15,batch_size=128)
test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X,y,epochs=15,batch_size=128)

data_oldtown = pd.read_csv('oldtown.csv',header=0,sep=";")
X_oldtown = np.array(data_oldtown, dtype = float)
predictions = model.predict(X_oldtown)
print(X_oldtown)

print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))
print(model.predict_classes(X_oldtown))