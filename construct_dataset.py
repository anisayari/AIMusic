import pandas as pd
import csv
from tqdm import tqdm
import youtube_dl
import pylast
import os
import googleapiclient.discovery
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import librosa.display
import uuid
import sklearn
import numpy as np
import time
import urllib.request
from bs4 import BeautifulSoup


API_KEY_lastfm = ""
API_SECRET_lastfm = ""
username_lastfm = ""
DEVELOPER_KEY_GCP = ""



def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def create_waveform_image(x,sr,uuid):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.savefig('data/images/waveform/waveform_{}.png'.format(uuid))
    plt.clf()

def create_spectrogram_image(x,sr,uuid):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.savefig('data/images/spectogram/spectrogram_{}.png'.format(uuid))
    plt.clf()


def extract_features_from_a_song(x,sr):
    dict_features = {
        'zcr': np.mean(librosa.feature.zero_crossing_rate(x)),
        'chroma_stft': np.mean(librosa.feature.chroma_stft(x, sr=sr)),
        'mfcc': np.mean(librosa.feature.mfcc(x, sr=sr)),
        'spec_cent': np.mean(librosa.feature.spectral_centroid(x, sr=sr)),
        'spec_bw': np.mean(librosa.feature.spectral_bandwidth(x, sr=sr)),
        'spec_rolloff': np.mean(librosa.feature.spectral_centroid(x, sr=sr))
    }
    return dict_features

def get_data_from_sound(row):
    dict_features = {}
    dict_features['chroma_stft'] = 0
    dict_features['mfcc'] = 0
    dict_features['spec_cent'] = 0
    dict_features['spec_rolloff'] = 0
    dict_features['spec_bw'] = 0
    dict_features['zcr'] = 0
    audio_path = 'data/music/{}/{}.mp3'.format(row['style'],str(row['uuid']))
    exists = os.path.isfile(audio_path)
    if exists and row['spec_cent'] == 0:
        x, sr = librosa.load(audio_path)
        uuid = row['uuid']
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
            dict_features = extract_features_from_a_song(x,sr)
        except OverflowError:
            print('Cannot run features extract for',uuid)
        print('{} DONE'.format(uuid))
        row['spec_cent'], row['spec_rolloff'], \
        row['spec_bw'], row['mfcc'], row['zcr'], row['chroma_stft'] = dict_features['spec_cent'], \
                                                                      dict_features['spec_rolloff'], \
                                                                      dict_features['spec_bw'], \
                                                                      dict_features['mfcc'], \
                                                                      dict_features['zcr'], \
                                                                      dict_features['chroma_stft']
    return row

def get_youtube_music(row):
    path = 'data/music/{}/{}.mp3'.format(row['style'],str(row['uuid']))
    exists = os.path.isfile(path)
    if exists:
        print('Already exist {}'.format(row['uuid']))
        return
    if row['videoID_youtube'] != "missing":
        try:
            videoID = row['videoID_youtube']
            ydl_opts = {
                'outtmpl': path,
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3'
                }]
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(['https://www.youtube.com/watch?v={}'.format(videoID)])
        except:
            print('Cannot find Youtube Video for {}'.format(str(row['uuid'])))

def build_dataset_by_tag(tag_list,output_file, df=None,df_passed=False):
    print('[INFO] Bulding Dataset by Tag....')
    network_lastfm = pylast.LastFMNetwork(api_key=API_KEY_lastfm, api_secret=API_SECRET_lastfm,username=username_lastfm)
    if not df_passed:
        df = pd.DataFrame( columns=['uuid','style','artist','title_music','lastfm_music_url','lastfm_artist_url','videoID_youtube'])
    tqdm.pandas()
    for style in tag_list:
        dict_to_add = {
            'style': style,
        }
        tracks = network_lastfm.get_tag(style).get_top_tracks(limit=1000)
        for track in tracks:
            if df_passed:
                if (str(track.item.get_artist()) in df['artist'].tolist()) and (str(track.item.get_name()) in df['title_music'].tolist()):
                    print('{} already exist'.format(str(track.item.get_artist())+'-'+str(track.item.get_name())))
                    continue
            dict_to_add['artist'] = track.item.get_artist()
            dict_to_add['title_music'] = track.item.get_name()
            dict_to_add['lastfm_music_url'] = track.item.get_artist().get_url()
            dict_to_add['lastfm_artist_url'] = track.item.get_url()
            dict_to_add['videoID_youtube'] = "missing"
            dict_to_add['uuid'] =  str(uuid.uuid4())[:8]
            dict_to_add['chroma_stft'] = 0
            dict_to_add['mfcc'] = 0
            dict_to_add['spec_cent'] = 0
            dict_to_add['spec_rolloff'] = 0
            dict_to_add['spec_bw'] = 0
            dict_to_add['zcr'] = 0
            df = df.append(dict_to_add , ignore_index=True)
    df.to_csv(output_file, sep=";", header=True, index=False)
    print('[INFO] Bulding Dataset by Tag DONE')

def get_video_id_from_parser(df):
    def get_info_from_parse(row):
        if row['videoID_youtube'] == "missing":
            query = str(row['artist'] +' '+row['title_music'])
            query = urllib.parse.quote(query)
            url = "https://www.youtube.com/results?search_query=" + query
            try:
                response = urllib.request.urlopen(url)
                html = response.read()
                soup = BeautifulSoup(html, 'html.parser')
                video_id_list = []
                for vid in soup.findAll(attrs={'class': 'yt-uix-tile-link'}):
                    if 'watch' in vid['href']:
                        video_id_list.append(vid['href'].replace('/watch?v=', ''))
            except urllib.error.HTTPError:
                print('Cannot Access the Request')
                return row
            videoID = video_id_list[0]
            row['videoID_youtube'] = videoID
            print('VideoID ok for {}'.format(str(row['artist'] + ' ' + row['title_music'])))
            with open('data/videoid.csv', 'a',newline='') as writeFile:
                writer = csv.writer(writeFile)
                try:
                    writer.writerow([row['artist'], row['title_music'], videoID])
                except:
                    print('[ERROR] Cannot write in csv video ID')
        else:
            print('VideoID already exist {}'.format(str(row['artist'] +' '+row['title_music'])))
        return row
    tqdm.pandas()
    df = df.progress_apply(get_info_from_parse, axis=1)
    return df

def get_video_id_from_youtube(df):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY_GCP)
    def get_info_from_youtube(row):
        print(str(row['artist'] +' '+row['title_music']))
        if row['videoID_youtube'] == "missing":
            query = str(row['artist'] +' '+row['title_music'])
            request = youtube.search().list(
            part="snippet",
            maxResults=1,
            q= query
            )
            try:
                response = request.execute()
                time.sleep(1)
            except googleapiclient.errors.HttpError:
                print('Cannot Access the Youtube API')
                return row
            videoID = response['items'][0]['id']['videoId']
            row['videoID_youtube'] = videoID
            with open('data/videoid.csv', 'a',newline='') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow([row['artist'], row['title_music'], videoID])
        else:
            print('VideoID already exist {}'.format(str(row['artist'] +' '+row['title_music'])))
        return row
    tqdm.pandas()
    df = df.progress_apply(get_info_from_youtube,axis=1)
    return df

output_file ='data/tracks_list.csv'
#df = pd.read_csv(output_file,sep=";", header=0)
#build_dataset_by_tag(['country','rap'], output_file,df,df_passed=True)
df = pd.read_csv(output_file,sep=";", header=0)
#df = get_video_id_from_youtube(df)
#df = get_video_id_from_parser(df)
#df.to_csv(output_file, sep=";", header=True, index=False)
tqdm.pandas()
#DOWNLOAD VIDEO FROM YOUTUBE
df.progress_apply(get_youtube_music, axis=1)
#EXTRACT FEATURES FROM SONG
tqdm.pandas()
df = df.progress_apply(get_data_from_sound , axis=1)
df.to_csv(output_file, sep=";", header=True, index=False)