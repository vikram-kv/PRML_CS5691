'''
driver code for ANN
-------------------
Author - Vedant (CS19B074), Vikram (CS19B021)
'''

from operator import le
import matplotlib.pyplot as plt
from inputmodule import *
from plotmodule import *
import ldamodule, pcamodule
import argparse
import shutil, os
from ANN import ANN
from sklearn.metrics import accuracy_score

colors = ['red','brown','lime','gold','dodgerblue']         # for plotting comparative ROC/DET curves

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='transformation to apply : \'none\' for None, \'pca\' for PCA and \'lda\' for LDA', type=str, required=True, dest='transform', choices=['none','pca','lda'])
    parser.add_argument('-d', help='Data type : \'synthetic\' for Synthetic Dataset, \'image\' for Image Dataset, \'spoken_digit\' for Isolated Spoken Digits and \'handwritten\' for Handwritten Character Dataset', type=str, required=True, dest='type', choices=['synthetic','image','spoken_digit','handwritten'])
    args = parser.parse_args()

    folder_suffix = '_ANN'
    if(args.type == 'synthetic'):
        (trainData, trainLabels, devData, devTrueLabels, _) = inputSYN()
        folder_suffix += '_SYN'

    elif(args.type == 'image'):
        ch = int(input('Enter 1 if you want to normalize. Else, enter 0 :'))
        (trainData, trainLabels, devData, devTrueLabels, _) = inputIMG(norm=ch)
        folder_suffix += '_IMG'

    elif(args.type == 'spoken_digit'):
        (trainData, trainLabels, devData, devTrueLabels, _) = inputISD()
        folder_suffix += '_ISD'

    elif(args.type == 'handwritten'):
        (trainData, trainLabels, devData, devTrueLabels, _) = inputHCD()
        folder_suffix += '_HCD'

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
    
    
