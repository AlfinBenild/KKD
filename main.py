#!/usr/bin/env python

import numpy as np
import pyaudio
import wave 
from pydub.playback import play
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import get_window
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
from time import sleep
from mfrc522 import SimpleMFRC522
import wiringpi as wiringpi

# Pin and variable definitions

buzzer = 40
switch_record = 38
switch_seat_belt = 36
mq3 = 26
red = 37
green = 35
voice = 38
sos = 33
power = 32
authentication = 31

GPIO.setmode(GPIO.BOARD)
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setup(switch_record, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(switch_seat_belt, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(sos, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(voice, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(power, GPIO.OUT)
GPIO.setup(authentication, GPIO.OUT)
GPIO.output(authentication, GPIO.LOW)

GPIO.output(power, GPIO.LOW)

commands_to_ids = {'Down' : 0, 'Engine' : 1, 'Off' : 2, 'On' : 3, 'One' : 4, 'Three' : 5, 'Two' : 6, 'Up' : 7, 'Window' : 8, 'Wiper' : 9}
ids_to_commands = {0 : 'Down', 1: 'Engine', 2 : 'Off', 3 : 'On', 4 : 'One', 5 : 'Three', 6 : 'Two',  7 : 'Up', 8 : 'Window', 9 : 'Wiper'}

interpreter = tflite.Interpreter("voice_model_softmax.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
GPIO.output(power, GPIO.HIGH)

reader = SimpleMFRC522()
hash_key = "68c07b6bf4f591095ad1c43c065a801822a6c9cdd8e15364"
id1 = "702537013584"
id2 = "807655716992"
id, hash_val = reader.read()
wiringpi.wiringPiSetupGpio()
wiringpi.pinMode(mq3, 0)
GPIO.setup(red, GPIO.OUT)
GPIO.setup(green, GPIO.OUT)

# Conditions
id = str(id)
rfid_key = (hash_key == str(hash_val)) and (id == id1 or id == id2)
seat_belt = GPIO.input(switch_seat_belt)
alcohol = wiringpi.digitalRead(mq3)

# Functions

def record(command):
    # Record in chunks of 1024 samples
    chunk = 1024
    # 16 bits per sample
    sample_format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000
    seconds = 1.5
    filename = "{}.wav".format(command)

    # Create an interface to PortAudio
    pa = pyaudio.PyAudio()
    stream = pa.open(format = sample_format, channels = channels,
                    rate = sample_rate, input = True, frames_per_buffer = chunk)
    print("Recording...")
    # Initialize array that be used for storing frames
    frames = [] 

    for i in range(0, int(sample_rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    # Terminate - PortAudio interface
    pa.terminate()

    print("Done !!!")

    # Save the recorded data in a .wav format
    sf = wave.open(filename, 'wb')
    sf.setnchannels(channels)
    sf.setsampwidth(pa.get_sample_size(sample_format))
    sf.setframerate(sample_rate)
    sf.writeframes(b''.join(frames))
    sf.close()
    return filename

def normalize_audio(audio):
    audio = audio/np.max(np.abs(audio))
    return audio

def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=16000):
    # hop_size in ms
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=16000):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis


def preprocess(filepath):
    sample_rate, audio = wavfile.read(filepath)
    audio = normalize_audio(audio)

    hop_size = 15 #ms
    FFT_size = 2048

    audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
    window = get_window("hann", FFT_size, fftbins=True)
    audio_win = audio_framed * window
    audio_winT = np.transpose(audio_win)

    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

    audio_fft = np.transpose(audio_fft)
    audio_power = np.square(np.abs(audio_fft))
    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 10
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=16000)
    filters = get_filters(filter_points, FFT_size)
    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]
    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)
    dct_filter_num = 40

    dct_filters = dct(dct_filter_num, mel_filter_num)
    cepstral_coefficents = np.dot(dct_filters, audio_log)
    return cepstral_coefficents

def buzz_bw_commands():
    GPIO.output(buzzer, GPIO.HIGH)
    sleep(0.5)
    GPIO.output(buzzer, GPIO.LOW)


while (True):
    id, hash_val = reader.read()
    id = str(id)
    hash_val = str(hash_val)
    rfid_key = (hash_key == hash_val)  and (id == id1 or id == id2)
    seat_belt = GPIO.input(switch_seat_belt)
    alcohol = wiringpi.digitalRead(mq3)
    print(rfid_key, seat_belt, not alcohol)
    if rfid_key and seat_belt and not alcohol:
        break

print("Authentication Successful !!")
GPIO.output(authentication, GPIO.HIGH)
while True:
    if GPIO.input(sos) == GPIO.HIGH:
        pass

    if GPIO.input(voice) == GPIO.HIGH:
        command1 = record('voice1')
        # Put some delay as of now input is used
        buzz_bw_commands()
        command2 = record('voice2')
        word1 = preprocess(command1).reshape(1, 40, 99, 1).astype(np.float32)
        word2 = preprocess(command2).reshape(1, 40, 99, 1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], word1)
        interpreter.invoke()
        word1_prob = interpreter.get_tensor(output_details[0]['index'])
        word1_prob = word1_prob.squeeze() * np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1])
        word1_pred = ids_to_commands[np.argmax(word1_prob)]

        interpreter.set_tensor(input_details[0]['index'], word2)
        interpreter.invoke()
        word2_prob = interpreter.get_tensor(output_details[0]['index'])

        commands = ['Down', 'Engine', 'Off', 'On', 'One', 'Three', 'Two', 'Up', 'Window', 'Wiper']

        if word1_pred == "Engine":
            word2_prob = word2_prob.squeeze() * np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        elif word1_pred == "Window":
            word2_prob = word2_prob.squeeze() * np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif word1_pred == "Wiper":
            word2_prob = word2_prob.squeeze() * np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])

        word2_pred = ids_to_commands[np.argmax(word2_prob)]
        command_word = word1_pred + " " + word2_pred
        print(command_word)

        if command_word == "Engine On":
            GPIO.output(green, GPIO.HIGH)
        if command_word == "Engine Off":
            GPIO.output(green, GPIO.LOW)
        if command_word == "Window Up":
            GPIO.output(red, GPIO.HIGH)
        if command_word == "Window Down":
            GPIO.output(red, GPIO.LOW)


