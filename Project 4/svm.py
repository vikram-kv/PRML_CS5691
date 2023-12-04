# main module for SVM classifier
# PRML A4 - Vikram CS19B021, Vedant Saboo CS19B074

import matplotlib.pyplot as plt
from inputmodule import *
from plotmodule import *
import transformmodule
import argparse
import os
from sklearn import svm
from sklearn.metrics import accuracy_score

colors = ['red','brown','lime','gold','dodgerblue']         # for plotting comparative ROC/DET curves

if __name__ == '__main__':
    # definitions of a few commandline args for data type, transform type and number of diff kernel types to pick
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='transformation to apply : \'none\' for None, \'pca\' for PCA and \'lda\' for LDA', type=str, required=True, dest='transform', choices=['none','pca','lda'])
    parser.add_argument('-d', help='Data type : \'synthetic\' for Synthetic Dataset, \'image\' for Image Dataset, \'spoken_digit\' for Isolated Spoken Digits and \'handwritten\' for Handwritten Character Dataset', type=str, required=True, dest='type', choices=['synthetic','image','spoken_digit','handwritten'])
    parser.add_argument('-n',help='Number of cases = different kernels to compare (max 5)', type=int, default=1, dest='ncases', choices=[1,2,3,4,5])
    args = parser.parse_args()

    # if-else block to extract data based on its type = args.type and modify folder_suffix accordingly
    folder_suffix = '_SVM'
    if(args.type == 'synthetic'):
        (trainData, trainLabels, devData, devTrueLabels, _) = inputSYN()
        folder_suffix += '_SYN'

    elif(args.type == 'image'):
        # we allow the choice of normalization for image data alone
        ch = int(input('Enter 1 if you want to normalize. Else, enter 0 :'))
        (trainData, trainLabels, devData, devTrueLabels, _) = inputIMG(norm=ch)
        folder_suffix += '_IMG'

    elif(args.type == 'spoken_digit'):
        (trainData, trainLabels, devData, devTrueLabels, _) = inputISD()
        folder_suffix += '_ISD'

    elif(args.type == 'handwritten'):
        (trainData, trainLabels, devData, devTrueLabels, _) = inputHCD()
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
    
    # get the different kernels to try out
    kernels = []
    for i in range(args.ncases):
        knl = input('Enter kernel {}: '.format(i+1))
        if(knl == 'poly'):
            deg = int(input("Enter the degree for 'poly' kernel: "))
            kernels.append((knl, deg))
        else:
            kernels.append((knl,0))         # to maintain uniformity, we use a dummy degree = 0 here. non 'poly' kernels will ignore it anyways
    
    # create the outfolder to save the plots
    outfolder = "plots"
    outfolder += folder_suffix
    while os.path.exists(outfolder):
        outfolder += " copy"
    os.mkdir(outfolder)

    fig1, roc = plt.subplots(figsize=(5,5), num=1)    # figure for ROC
    fig2, det = plt.subplots(figsize=(5,5), num=2)    # figure for DET

    # loop to try out the different kernel types
    for i in range(len(kernels)):
        knl, deg = kernels[i]

        # create a SVM object with kernel = knl and degree = deg, fit the train data in it,
        # and the scores of dev data from it. non 'poly' kernels ignore degree parameter
        clf = svm.SVC(kernel=knl, degree=deg, probability=True, random_state=12345678)
        clf.fit(trainData, trainLabels)
        classes = clf.classes_
        probs = clf.predict_log_proba(devData)

        # we make prediction as argmax of scores for each test vector
        predLabels = []
        for j in range(len(probs)):
            predLabels.append(classes[np.argmax(probs[j])])

        # name for the current kernel. used in labelling plots
        name = knl
        if(knl == 'poly'):
            name += '_degree={}'.format(deg)

        # print the accuracy
        print('kernel = {} : {:.2f}'.format(name, accuracy_score(devTrueLabels, predLabels)*100))

        # save the confusion matrix for the current kernel 
        _, cfm = plt.subplots(figsize=(5,5),num=(i+3))
        cfm = ConfusionMatrixPlotter(classlist=classes,truth=devTrueLabels,pred=predLabels,ax=cfm)
        plt.savefig(outfolder+'/ConfusionMatrix_kernel={}.png'.format(name))

        # construct the scores dictionary from class C to the list of scores of each test vectors
        # towards the class C and use it to make the ROC/DET plots
        scores = dict()
        for j in range(len(classes)):
            scores[classes[j]] = probs[:,j]

        roc = ROC_curve(truth=devTrueLabels,scorelists=scores,ax=roc,clr=colors[i],lbl='kernel={}'.format(name))
        det = DET_curve(truth=devTrueLabels,scorelists=scores,ax=det,clr=colors[i],lbl='kernel={}'.format(name))

    # code to save the comparative ROC and DET plots in outfolder
    plt.figure(num=1)
    roc.legend(fontsize=8)
    roc.set_title(' Receiver Operating Characteristic ', fontsize=10)
    plt.savefig(outfolder+'/ReceiverOperatingCharacteristic.png')
    plt.figure(num=2)
    det.legend(fontsize=8)
    det.set_title(' Detection Error Tradeoff ', fontsize=10)
    plt.savefig(outfolder+'/DetectionErrorTradeoff.png')