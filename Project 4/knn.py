# main module for KNN classifier
# PRML A4 - Vikram CS19B021, Vedant Saboo CS19B074

import matplotlib.pyplot as plt
from inputmodule import *
from plotmodule import *
import transformmodule, knnmodule
import argparse
import os
from sklearn.metrics import accuracy_score

colors = ['red','brown','lime','gold','dodgerblue']         # for plotting comparative ROC/DET curves

if __name__ == '__main__':
    # definitions of a few commandline args for data type, transform type and number of diff K values to consider
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='transformation to apply : \'none\' for None, \'pca\' for PCA and \'lda\' for LDA', type=str, required=True, dest='transform', choices=['none','pca','lda'])
    parser.add_argument('-d', help='Data type : \'synthetic\' for Synthetic Dataset, \'image\' for Image Dataset, \'spoken_digit\' for Isolated Spoken Digits and \'handwritten\' for Handwritten Character Dataset', type=str, required=True, dest='type', choices=['synthetic','image','spoken_digit','handwritten'])
    parser.add_argument('-n',help='Number of cases = different values of k in KNN to compare (max 5)', type=int, default=1, dest='ncases', choices=[1,2,3,4,5])
    args = parser.parse_args()

    # if-else block to extract data based on its type = args.type and modify folder_suffix accordingly
    folder_suffix = '_KNN'
    if(args.type == 'synthetic'):
        (trainData, trainLabels, devData, devTrueLabels, classlist) = inputSYN()
        folder_suffix += '_SYN'

    elif(args.type == 'image'):
        # we allow the choice of normalization for image data alone
        ch = int(input('Enter 1 if you want to normalize. Else, enter 0 :'))
        (trainData, trainLabels, devData, devTrueLabels, classlist) = inputIMG(norm=ch)
        folder_suffix += '_IMG'

    elif(args.type == 'spoken_digit'):
        (trainData, trainLabels, devData, devTrueLabels, classlist) = inputISD()
        folder_suffix += '_ISD'

    elif(args.type == 'handwritten'):
        (trainData, trainLabels, devData, devTrueLabels, classlist) = inputHCD()
        folder_suffix += '_HCD'

    # if-else block to transform data based on args.transform and modify folder_suffix accordingly
    if(args.transform == 'none'):
        folder_suffix += '_NOTRANSFORM'
    
    elif(args.transform == 'pca'):
        n_comp = int(input('Enter the dimension required after PCA: '))
        pca = transformmodule.PCA(n_comp)
        pca.fit(trainData)
        trainData = pca.transform_data(trainData)
        devData = pca.transform_data(devData)
        folder_suffix += '_PCATRANSFORM'
    
    elif(args.transform == 'lda'):
        n_comp = int(input('Enter the dimension required after LDA: '))
        lda = transformmodule.FDA(trainData, trainLabels, n_comp)
        trainData = lda.transform_data(trainData)
        devData = lda.transform_data(devData)
        folder_suffix += '_LDATRANSFORM'
    
    # create the outfolder to save the plots
    outfolder = "plots"
    outfolder += folder_suffix
    while os.path.exists(outfolder):
        outfolder += " copy"
    os.mkdir(outfolder)
    
    # get args.ncases diff K values to try for the KNN classifier
    l = input('Enter the k values separated by spaces: ')
    klist = l.strip().split()
    klist = list(map(int,klist))
    if(len(klist) != args.ncases):
        print('Illegal Usage - Enter only {} values'.format(args.ncases))
        exit(-1)

    fig1, roc = plt.subplots(figsize=(5,5), num=1)    # figure for ROC
    fig2, det = plt.subplots(figsize=(5,5), num=2)    # figure for DET

    # loop to try out the different K values
    for i in range(0, args.ncases):
        kval = klist[i]
        knnclassifier = knnmodule.KNN(num_neighbours=kval)                                              # create KNN object with K = kval
        knnclassifier.fit(trainData, trainLabels, classlist)                                            # add the train points to the KNN object
        predLabels = knnclassifier.predict(devData)                                                     # classify the test points to diff classes
        print('k = {} : {:.2f}'.format(kval, accuracy_score(devTrueLabels, predLabels)*100))            # print accuracy

        # code to plot confusion matrix and save the figure in outfolder
        _, cfm = plt.subplots(figsize=(5,5),num=(i+3))
        cfm = ConfusionMatrixPlotter(classlist=classlist,truth=devTrueLabels,pred=predLabels,ax=cfm)
        plt.savefig(outfolder+'/ConfusionMatrix_k={}.png'.format(kval))

        # get the scores dictionary from class C to list of each test vector score towards the class C
        # and plot the ROC/DET curves using it
        scores = knnclassifier.predictscores(testvectors=devData)           
        roc = ROC_curve(truth=devTrueLabels,scorelists=scores,ax=roc,clr=colors[i],lbl='k={}'.format(kval))
        det = DET_curve(truth=devTrueLabels,scorelists=scores,ax=det,clr=colors[i],lbl='k={}'.format(kval))

    # code to save the comparative ROC and DET plots in outfolder
    plt.figure(num=1)
    roc.legend(fontsize=8)
    roc.set_title(' Receiver Operating Characteristic ', fontsize=10)
    plt.savefig(outfolder+'/ReceiverOperatingCharacteristic.png')
    plt.figure(num=2)
    det.legend(fontsize=8)
    det.set_title(' Detection Error Tradeoff ', fontsize=10)
    plt.savefig(outfolder+'/DetectionErrorTradeoff.png')