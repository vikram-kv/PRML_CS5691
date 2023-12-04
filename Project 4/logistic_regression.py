from operator import le
import matplotlib.pyplot as plt
from inputmodule import *
from plotmodule import *
import transformmodule
import argparse
import shutil, os
from LogisticRegression import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

colors = ['red','brown','lime','gold','dodgerblue']         # for plotting comparative ROC/DET curves

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='transformation to apply : \'none\' for None, \'pca\' for PCA and \'lda\' for LDA', type=str, required=True, dest='transform', choices=['none','pca','lda'])
    parser.add_argument('-d', help='Data type : \'synthetic\' for Synthetic Dataset, \'image\' for Image Dataset, \'spoken_digit\' for Isolated Spoken Digits and \'handwritten\' for Handwritten Character Dataset', type=str, required=True, dest='type', choices=['synthetic','image','spoken_digit','handwritten'])
    args = parser.parse_args()

    folder_suffix = '_LR'
    if(args.type == 'synthetic'):
        (trainData, trainLabels, devData, devTrueLabels, classes) = inputSYN()
        folder_suffix += '_SYN'

    elif(args.type == 'image'):
        ch = int(input('Enter 1 if you want to normalize. Else, enter 0 :'))
        (trainData, trainLabels, devData, devTrueLabels, classes) = inputIMG(norm=ch)
        folder_suffix += '_IMG'

    elif(args.type == 'spoken_digit'):
        (trainData, trainLabels, devData, devTrueLabels, classes) = inputISD()
        folder_suffix += '_ISD'

    elif(args.type == 'handwritten'):
        (trainData, trainLabels, devData, devTrueLabels, classes) = inputHCD()
        folder_suffix += '_HCD'

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
    
    kernels = ['linear', 'rbf', 'sigmoid']
    
    # outfolder = input('Enter a name. An appropriate suffix will be added to get the folder name in which the plots will be saved: ')
    outfolder = "plots"
    outfolder += folder_suffix
    while os.path.exists(outfolder):
        # shutil.rmtree(outfolder)
        outfolder +=" copy"
    os.mkdir(outfolder)

    fig1, roc = plt.subplots(figsize=(5,5), num=1)    # figure for ROC
    fig2, det = plt.subplots(figsize=(5,5), num=2)    # figure for DET

    mode = 'binary' if args.type == 'synthetic' else 'multi_class'
    LRMachine = LR(mode=mode)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(trainLabels)
    encodedTrainLabels = encoder.transform(trainLabels)
    LRMachine.fit(trainData, encodedTrainLabels)
    probs = LRMachine.predict_proba(devData)
    predLabels = []

    for j in range(len(probs)):
        if(mode=='binary'):
            predLabels.append(classes[int(probs[j] > 0.5)])
        else:
            predLabels.append(classes[np.argmax(probs[j])])

    scores = dict()
    for j in range(len(classes)):
        if(mode=='binary'):
            scores[classes[j]] = probs if j == 1 else 1 - probs
        else:
            scores[classes[j]] = probs[:,j]

    _, cfm = plt.subplots(figsize=(5,5))
    cfm = ConfusionMatrixPlotter(classlist=classes,truth=devTrueLabels,pred=predLabels,ax=cfm)

    plt.savefig(outfolder+'/ConfusionMatrix.png')
    roc = ROC_curve(truth=devTrueLabels,scorelists=scores,ax=roc,clr='blue')
    det = DET_curve(truth=devTrueLabels,scorelists=scores,ax=det,clr='blue')

    plt.figure(num=1)
    roc.legend(fontsize=8)
    roc.set_title(' Receiver Operating Characteristic ', fontsize=10)
    plt.savefig(outfolder+'/ReceiverOperatingCharacteristic.png')
    plt.figure(num=2)
    det.legend(fontsize=8)
    det.set_title(' Detection Error Tradeoff ', fontsize=10)
    plt.savefig(outfolder+'/DetectionErrorTradeoff.png')