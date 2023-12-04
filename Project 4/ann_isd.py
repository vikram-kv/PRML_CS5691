# -*- coding: utf-8 -*-
import os
import time
import random
import tflearn
import librosa
import numpy as np
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

from torch import classes
from inputmodule import inputISD
import pcamodule, ldamodule

#
# EXTRACT MFCC FEATURES
#
def extract_mfcc(file_path, utterance_length):
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)
    
    return mfcc_features

'''
Minibatch generator function
'''
def get_mfcc_batch(train_data, train_labels, batch_size:int = 20, utterance_length:int=76):
    ft_batch = []
    label_batch = []
    indices = [x for x in range(len(train_data))]

    while True:
        # Shuffle Files
        random.shuffle(indices)
        for i in indices:
            
            mfcc_features = train_data[i].reshape(-1,utterance_length)
            
            # One-hot encode label for 5 classes
            label = np.eye(5)[int(train_labels[i])]
            
            # Append to label batch
            label_batch.append(label)
            
            # Append mfcc features to ft_batch
            ft_batch.append(mfcc_features)

            # Check to see if default batch size is < than ft_batch
            if len(ft_batch) >= batch_size:
                # send over batch
                yield ft_batch, label_batch
                # reset batches
                ft_batch = []
                labels_batch = []

'''
Display function (not needed for classifier)
'''
def display_power_spectrum(wav_file_path, utterance_length):
    mfcc = extract_mfcc(wav_file_path, utterance_length)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.show()

    # Feature information
    print("Feature Shape: ", mfcc.shape)
    print("Features: " , mfcc[:,0])

"""
MAIN
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='transformation to apply : \'none\' for None, \'pca\' for PCA and \'lda\' for LDA', type=str, required=True, dest='transform', choices=['none','pca','lda'])
    # parser.add_argument('-d', help='Data type : \'synthetic\' for Synthetic Dataset, \'image\' for Image Dataset, \'spoken_digit\' for Isolated Spoken Digits and \'handwritten\' for Handwritten Character Dataset', type=str, required=True, dest='type', choices=['synthetic','image','spoken_digit','handwritten'])
    args = parser.parse_args()

    folder_suffix = '_ANN'
    # if(args.type == 'synthetic'):
    #     (trainData, trainLabels, devData, devTrueLabels, _) = inputSYN()
    #     folder_suffix += '_SYN'

    # elif(args.type == 'image'):
    #     ch = int(input('Enter 1 if you want to normalize. Else, enter 0 :'))
    #     (trainData, trainLabels, devData, devTrueLabels, _) = inputIMG(norm=ch)
    #     folder_suffix += '_IMG'

    # elif(args.type == 'spoken_digit'):
    (trainData, trainLabels, devData, devTrueLabels, _) = inputISD()
    transform_dict = {'1':0,'4':1,'6':2,'o':3,'z':4}
    retransform_dict = {0:'1',1:'4',2:'6',3:'o',4:'z'}
    train_labels_encoded = [transform_dict[x] for x in trainLabels]
    dev_labels_encoded = [transform_dict[x] for x in devTrueLabels]
    folder_suffix += '_ISD'

    # elif(args.type == 'handwritten'):
    #     (trainData, trainLabels, devData, devTrueLabels, _) = inputHCD()
    #     folder_suffix += '_HCD'

    if(args.transform == 'none'):
        folder_suffix += '_NOTRANSFORM'
    
    elif(args.transform == 'pca'):
        n_comp = int(input('Enter the dimension required after PCA: '))
        pca = pcamodule.PCA(n_comp)
        pca.fit(trainData)
        trainData = pca.transform_data(trainData)
        devData = pca.transform_data(devData)
        folder_suffix += '_PCATRANSFORM'
    
    elif(args.transform == 'lda'):
        n_comp = int(input('Enter the dimension required after LDA: '))
        lda = ldamodule.FDA(trainData, trainLabels, n_comp)
        trainData = lda.transform_data(trainData)
        devData = lda.transform_data(devData)
        folder_suffix += '_LDATRANSFORM'
        
    # outfolder = input('Enter a name. An appropriate suffix will be added to get the folder name in which the plots will be saved: ')
    outfolder = "plots"
    outfolder += folder_suffix
    while os.path.exists(outfolder):
        # shutil.rmtree(outfolder)
        outfolder +=" copy"
    os.mkdir(outfolder)
    
    # Initial Parameters
    lr = 0.0001
    iterations_train = 1
    bsize = 195
    audio_features = 35  
    utterance_length = 76  # Modify to see what different results you can get
    # mfcc_feature_vector = 2660
    ndigits = 5

    # Divide training data in batches
    train_batch = get_mfcc_batch(train_data=trainData, train_labels=train_labels_encoded, batch_size=bsize, utterance_length=utterance_length)
    test_batch = get_mfcc_batch(train_data=devData, train_labels=dev_labels_encoded, batch_size=len(devData), utterance_length=utterance_length)
    
    
    # # Build Model
    sp_network = tflearn.input_data([None, audio_features, utterance_length])
    # LSTM or LONG SHORT TERM MEMORY IS A TYPE OF RNN THAT PROCESSES SEQUENTIAL DATA
    sp_network = tflearn.lstm(sp_network, 128*8, dropout=0.5)
    sp_network = tflearn.fully_connected(sp_network, ndigits, activation='softmax')
    sp_network = tflearn.regression(sp_network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy')
    sp_model = tflearn.DNN(sp_network, tensorboard_verbose=0)

    # Train Model
    while iterations_train > 0:
        X_tr, y_tr = next(train_batch)
        X_test, y_test = next(test_batch)
        sp_model.fit(X_tr, y_tr, n_epoch=100, validation_set=(X_test, y_test), show_metric=True, batch_size=bsize)
        iterations_train -=1
    if(not os.path.exists('model/')):
        os.mkdir('model')
    sp_model.save("model/speech_recognition.lstm")

    # Test Model
    sp_model.load('model/speech_recognition.lstm')
    # mfcc_features = extract_mfcc('../data/recordings/test/3_jackson_49.wav', utterance_length)
    # mfcc_features = mfcc_features.reshape((1,mfcc_features.shape[0],mfcc_features.shape[1]))
    # prediction_digit = sp_model.predict(mfcc_features)
    # print(prediction_digit)
    # print("Digit predicted: ", np.argmax(prediction_digit))

    # Done
    return 0


if __name__ == '__main__':
    main()