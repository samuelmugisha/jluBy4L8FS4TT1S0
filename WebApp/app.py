from flask import Flask, render_template, request, redirect, url_for
import os
from uvicorn import run
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

app = Flask(__name__)

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
classes = ['Synthetic', 'Real']
model = load_model("models/classifier.h5")
scalar = StandardScaler()

ALLOWED_EXT = set(['WAV', 'mp3', 'wav'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def htmloader(text,inputaudio,outputaudio):
    x = ""
    x+="<b>Utterance: </b><p>"+text+"</p><br>"
    x+="<b>Cloned Audio:</b><br>"
    x+="<audio controls>"
    x+="  <source src='"+str(outputaudio)+"' type='audio/wav'>"
    x+="</audio><br>"
    return x

def upload_file():
    fil = []
    filename = None
    if request.method == 'POST':
        file = request.files['audio_file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fil.append("File uploaded successfully")
            fil.append(os.path.join(UPLOAD_FOLDER, secure_filename(filename)))
        else:
            fil.append("Please upload an .mp3 or .wav file to synthesize audio")
    return fil


def prediction(filename, model):
    df = {'mfccs' : [], 'chroma' : [], 'mel' : [], 'contrast' : [], 'tonnetz' : []}
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    
    df['mfccs'].append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0))
    df['chroma'].append(np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(X)), sr=sample_rate).T,axis=0))
    df['mel'].append(np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0))
    df['contrast'].append(np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(X)), sr=sample_rate).T,axis=0))
    df['tonnetz'].append(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0))
    
    features_list = []
    for i in range(0, len(df['mfccs'])):
        features_list.append(np.concatenate((
        scalar.fit_transform((df['mfccs'][i]).reshape(-1, 1)),
        scalar.fit_transform((df['chroma'][i]).reshape(-1, 1)),
        scalar.fit_transform((df['mel'][i]).reshape(-1, 1)),
        scalar.fit_transform((df['contrast'][i]).reshape(-1, 1)),
        scalar.fit_transform((df['tonnetz'][i]).reshape(-1, 1))), axis=0))
    a = np.array(features_list).reshape(len(df['mfccs']), 193)
    
    prediction = model.predict(a, verbose=0)
    prediction_class = classes[int(tf.round(prediction))]
    return prediction_class, prediction    
    

            
@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/voice_cloning")
def voice_cloning():
    return render_template("voice_cloning.html")
    
    
@app.route("/fake_audio_detection")
def fad():
    return render_template("predict.html")
    
@app.route("/synthesize", methods=['GET', 'POST'])
def synthesize():
    uploaded = upload_file()
    utterance = None

    if request.method == 'POST':
        utterance = request.form['clone_text']
    
    if len(uploaded) == 1:
        return render_template("voice_cloning.html", error=uploaded[0])
        
    else:
        print(uploaded[1])
        from synthesizer.inference import Synthesizer
        from encoder import inference as encoder
        from vocoder import inference as vocoder
        from pathlib import Path
        import wavio

        encoder.load_model(Path("saved_models/default/encoder.pt"))
        synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
        vocoder.load_model(Path("saved_models/default/vocoder.pt"))
        
        n = 0
        
        audio_path = uploaded[1]
        original_audio, sample_rate = librosa.load(audio_path)
        embeddings = encoder.embed_utterance(encoder.preprocess_wav(audio_path, 22050))
        print("Synthesizing new audio...")
        specs = synthesizer.synthesize_spectrograms([str(utterance)], [embeddings])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        synthetic_path = 'static/output.wav'
        wavio.write(synthetic_path, generated_wav, sampwidth=2, rate=synthesizer.sample_rate)
        print("Saved output as ", synthetic_path)
        return render_template("voice_cloning.html", output=htmloader(str(utterance), audio_path, synthetic_path))
    
    return render_template("voice_cloning.html", error="Incorrect file")   

@app.route('/fake_audio', methods=['GET', 'POST'])
def fake_audio():
    uploaded = upload_file()
    if len(uploaded) == 1:
        return render_template("predict.html", error="Please upload .mp3 or .wav file")
    else:
        class_result , prob_result = prediction(filename=uploaded[1] , model=model)
        return render_template("predict.html", classes=class_result) 
        
    return "Ok"
    
    
if __name__=="__main__":
    app.run()