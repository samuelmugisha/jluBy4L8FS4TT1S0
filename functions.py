import os
from os.path import exists, join, basename, splitext

from pathlib import Path

# SV2TTS code from https://github.com/CorentinJ/Real-Time-Voice-Cloning, bundled in WebApp/
# with the pretrained models in WebApp/saved_models/default
project_name = Path(__file__).resolve().parent / 'WebApp'

import sys
sys.path.append(str(project_name))

from IPython.display import display, Audio, clear_output
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings("ignore")

scalar = StandardScaler()

# The voice cloning models (SV2TTS) and the speech/NLP libraries are only needed
# for Part 1, so they are loaded on first use. This lets the fake audio detection
# functions be used without the Real-Time-Voice-Cloning repo and pretrained models.
encoder = synthesizer = vocoder = nlp = None


def _load_voice_cloning():
    global encoder, synthesizer, vocoder
    if synthesizer is None:
        from synthesizer.inference import Synthesizer
        from encoder import inference as encoder
        from vocoder import inference as vocoder
        encoder.load_model(project_name / Path("saved_models/default/encoder.pt"))
        synthesizer = Synthesizer(project_name / Path("saved_models/default/synthesizer.pt"))
        vocoder.load_model(project_name / Path("saved_models/default/vocoder.pt"))


def _load_nlp():
    global nlp
    if nlp is None:
        import spacy
        nlp = spacy.load('en_core_web_sm')
    return nlp


def synthesize_audio(sample_audio, audio_text, audio_name):
    """
    Clones the voice of a given audio by performing text to voice conversion
    
    params:
    sample_audio: Audio file to use as a samlple for text to voice conversion
    audio_text:
    audio_name: 
    """
    import wavio
    _load_voice_cloning()
    embeddings = encoder.embed_utterance(encoder.preprocess_wav(sample_audio, 22050))
    print("Synthesizing new audio...")
    specs = synthesizer.synthesize_spectrograms([audio_text], [embeddings])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    clear_output()
    wavio.write(audio_name, generated_wav, sampwidth=2, rate=synthesizer.sample_rate)

    
def speech_to_text(audio):
    """
    Transcribes an audio file into text
    """
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(audio) as src:
        #r.adjust_for_ambient_noise(src)
        audio = r.record(src)
        text = r.recognize_google(audio, language='en-US')
        return text

    
def text_comparison(cloned_audio, original_text, nlp=None):
    """
    Returns Word Error Rate and similarity score between the texts 
    generated from the real audio file and synthetic audio.
    """
    import jiwer
    from jiwer import wer
    nlp = nlp or _load_nlp()
    cloned_text = speech_to_text(cloned_audio)
    transformation = jiwer.Compose([jiwer.ToLowerCase(),
                                    jiwer.RemoveWhiteSpace(replace_by_space=True),
                                    jiwer.Strip(),
                                    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")]) 
    word_error_rate = wer(reference=original_text, 
                          hypothesis=cloned_text, 
                          reference_transform=transformation, 
                          hypothesis_transform=transformation)
    print('cloned_text: ', cloned_text)
    print('original_text: ', original_text)
    print("####################################################################")
    return nlp(original_text).similarity(nlp(cloned_text)), word_error_rate


def concat_audio(files, file_name):
    """
    Function to combine the recordings from the same speaker into a single audio file
    """
    from moviepy.editor import concatenate_audioclips, AudioFileClip
    audio_files = [AudioFileClip(clip) for clip in files]
    final_recording = concatenate_audioclips(audio_files)
    final_recording.write_audiofile(file_name)
    
    
def extract_features(files):
    """
    Takes an audio file as the input and reaturns a dictionary with the following features for each audio file
    MFCC - Mel-Frequency Cepstral Coefficients
    Chromagram
    STFT - Short-Term Fourier Transform
    Spectral Contrasts
    Tonal centroid features
    """
    df = {'mfccs' : [], 'chroma' : [], 'mel' : [], 'contrast' : [], 'tonnetz' : []}
    for file_name in files:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        df['mfccs'].append(mfccs)
        
        stft = np.abs(librosa.stft(X))
        
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        df['chroma'].append(chroma)
        
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        df['mel'].append(mel)
        
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        df['contrast'].append(contrast)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        df['tonnetz'].append(tonnetz)
    return df


def plot_curves(model, epochs):
    """
    Plots Accuracy vs epochs and loss vs epochs curves after training
    """
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    epochs = min(epochs, len(model.history['loss']))
    ax[0].plot(range(epochs), model.history['accuracy'], label="Training Accuracy")
    ax[0].plot(range(epochs), model.history['val_accuracy'], label="Validation Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[1].plot(range(epochs), model.history['loss'], label="Training Loss")
    ax[1].plot(range(epochs), model.history['val_loss'], label="Validation Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    plt.show()
    

def concat_features(features):
    """
    Modifies the shape of the dataset, making it convinient to be used for training and testing
    """
    features_list = []
    for i in range(0, len(features['mfccs'])):
        features_list.append(np.concatenate((
            scalar.fit_transform((features['mfccs'][i]).reshape(-1, 1)),
            scalar.fit_transform((features['chroma'][i]).reshape(-1, 1)),
            scalar.fit_transform((features['mel'][i]).reshape(-1, 1)),
            scalar.fit_transform((features['contrast'][i]).reshape(-1, 1)),
            scalar.fit_transform((features['tonnetz'][i]).reshape(-1, 1))), axis=0))
    return np.array(features_list).reshape(len(features['mfccs']), 193)


def eval_results(actual, predictions):
    """
    Summarizes the results from training and testing phases in terms of accuracy, F-Score and confusion matrix
    """
    print("Accuracy of the model on test data : ", accuracy_score(actual, predictions))
    print("#"*100)
    print(classification_report(actual, predictions))
    cm = confusion_matrix(actual, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print("Total Synthetic recordings predicted", np.bincount(predictions.astype(int))[0])
    print("Total real recordings predicted", np.bincount(predictions.astype(int))[1])