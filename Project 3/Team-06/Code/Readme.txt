ASSIGNMENT #3
Vedant Saboo - CS19B074
K V Vikram   - CS19B021

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
Task A - GMM and K Means on Image Datset and Synthetic dataset

The main code is in main_gmm.py
The support files include : GMM.py which contains the class GMM
                          : KMeans.py which contains the class KMeans
                          : plotmodule.py which helps in plotting various graphs 

Run
$ python GMM.py
image/synthetic?
>> Type 'image' to run the model on Image dataset
>> Type 'synthetic' to run the model on Synthetic dataset

**requires**
        numpy library       : pip install numpy
        matplotlib library  : pip install matplotlib
        pandas library      : pip install pandas


----------------------------------------------------------------------------------------------------------------------------------------------------------------
Task B - DTW and HMM :- 

*) DTW :- 
    
    The main DTW code is in dtwmodule.py and plotting code is in plotmodule.py .
    We require numba library in order to speed up numpy-based python code by a large factor.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

    a) Isolated Spoken Digits(ISD):- (script is dtw_ISD.py)

        Place the folder F containing the ISD data in the code directory. F must contain a subdirectory for each class {1,4,6,o,z} and each subdirectory must
        have train/ and dev/ folders within them. All data files must have .mfcc extension.

        Execution : python3 dtw_ISD.py [YES | NO] [TOP_N] where the 1st commandline arg is YES if we want to use window(s) and NO otherwise. TOP_N is an integer
                    which will determine number of lowest scores to average to find the score of a test series for a class. 
        
        Some sample inputs and outputs are shown below assuming that ISD is the folder F and classes are 1, 4, 6, o, z.

        i) for cmd = $ python3 dtw_ISD.py NO 8

            I/O:-
            '''
                Enter the name of the folder containing the isolated digits data : ISD
                Enter the class names separated by spaces : 1 4 6 o z 
                Enter a string N. The plots will be saved in N_ISDDTW_Results: test1
            '''

            Result:-
                A folder 'test1_ISDDTW_Results' will be generated and it will have the confusion matrix, ROC plot and DET plot.

       ii) for cmd = $ python3 dtw_ISD.py YES 8

            I/O:-
            '''
                Enter the name of the folder containing the isolated digits data : ISD
                Enter the class names separated by spaces : 1 4 6 o z
                Enter the number of window sizes (at most 5).
                3 
                Suggestion : Values should be in [1-20] for good results.
                Enter the window sizes separated by spaces : 3 6 9
                Enter a string N. The plots will be saved in N_ISDDTW_Results: test2
            '''

            Result:-
                A folder 'test2_ISDDTW_Results' will be generated and it will have the confusion matrices for each window size, 
                a comparative ROC plot and a comparative DET plot.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

    b) Handwritten Character Data(HCD) without normalization (script is dtw_HCD_nonorm.py):-

        Place the folder F containing the HCD data in the code directory. F must contain a subdirectory for each class {a,ai,chA,dA,lA} and each subdirectory must
        have train/ and dev/ folders within them. All data files must have .txt extension.

        We note that no normalization will be done on the data before DTW.

        Execution : python3 dtw_HCD_nonorm.py [YES | NO] [TOP_N] where the 1st commandline arg is YES if we want to use window(s) and NO otherwise. TOP_N is an integer
                    which will determine number of lowest scores to average to find the score of a test series for a class. 
        
        Some sample inputs and outputs are shown below assuming that HCD is the folder F and classes are a, ai, chA, dA, lA.

        i) for cmd = $ python3 dtw_HCD_nonorm.py NO 15

            I/O:-
            '''
                Enter the name of the folder containing the handwritten char data : HCD
                Enter the class names separated by spaces : a ai chA dA lA
                Enter a string N. The plots will be saved in N_HCDDTW_Nonorm_Results: test3
            '''

            Result:-
                A folder 'test3_HCDDTW_Nonorm_Results' will be generated and it will have the confusion matrix, ROC plot and DET plot.

       ii) for cmd = $ python3 dtw_HCD_nonorm.py YES 15

            I/O:-
            '''
                Enter the name of the folder containing the handwritten char data : HCD
                Enter the class names separated by spaces : a ai chA dA lA
                Enter the number of window sizes (at most 5).
                5
                Suggestion : Values should be in [1-20] for good results.
                Enter the window sizes separated by spaces : 3 6 9 12 15
                Enter a string N. The plots will be saved in N_HCDDTW_Nonorm_Results: test4
            '''

            Result:-
                A folder 'test4_HCDDTW_Nonorm_Results' will be generated and it will have the confusion matrices for each window size, 
                a comparative ROC plot and a comparative DET plot.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

    c) Handwritten Character Data(HCD) with normalization (script is dtw_HCD_norm.py):-

            The instructions are the same as in (b) except that the character data will be normalized before DTW is performed. The script name
            is 'dtw_HCD_norm.py' and output folder will be suffixed by '_Norm_Results' now.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

**) HMM :-

    The main HMM helper functions are in hmmmodule.py, the K-Means code for vector quantisation is in hmmkmeansmodule.py, the plotting module used is
    plotmodule.py . We also require that the executables, train_hmm and test_hmm, which are obtained after making HMM-Code be placed in the code directory.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

    a) Isolated Spoken Digits(ISD) :- (script is hmm_ISD.py)

        Place the folder F containing the ISD data in the code directory. F must contain a subdirectory for each class {1,4,6,o,z} and each subdirectory must
        have train/ and dev/ folders within them. All data files must have .mfcc extension.

        Execution : python3 hmm_ISD.py 

        Sample input and output is shown below assuming that ISD is the folder F and classes are 1, 4, 6, o, z.

        i) for cmd = $ python3 hmm_ISD.py

            I/O:-
            '''
                Enter the name of the folder containing the isolated digits data : ISD
                Enter the class names separated by spaces : 1 4 6 o z
                Enter a name X. The plots will be saved in X_ISDHMM_Results: test5
                Enter the number of cases (at most 5):
                3
                Enter the M value for case 1 : 50
                Enter the N values for classes (in the order ['1', '4', '6', 'o', 'z']) for case 1 separated by spaces : 3 3 3 3 3
                Enter the M value for case 2 : 75 
                Enter the N values for classes (in the order ['1', '4', '6', 'o', 'z']) for case 2 separated by spaces : 5 5 3 3 3
                Enter the M value for case 3 : 80
                Enter the N values for classes (in the order ['1', '4', '6', 'o', 'z']) for case 3 separated by spaces : 5 5 5 5 5
                .....
            '''
                where ..... is output from train_hmm and test_hmm execution

            Result:-

                A folder 'test5_ISDHMM_Results' will be generated and it will have the confusion matrices, a comparative ROC plot and 
                a comparative DET plot. The hmm model (.hmm) files for the last case and the corresponding kmeans.codebook file 
                will also be saved along with the plots. These files are relevant for the optional part.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

    b) Handwritten Character Data(HCD) :- (script is hmm_HCD.py)

        Place the folder F containing the HCD data in the code directory. F must contain a subdirectory for each class {a,ai,chA,dA,lA} and each subdirectory must
        have train/ and dev/ folders within them. All data files must have .txt extension. 

        Execution : python3 hmm_HCD.py 

        Sample input and output is shown below assuming that HCD is the folder F and classes are a, ai, chA, dA, lA.

        i) for cmd = $ python3 hmm_HCD.py

            I/O:-
            '''
                Enter the name of the folder containing the handwritten char data : HCD
                Enter the class names separated by spaces : a ai chA dA lA
                Enter a name X. The plots will be saved in X_HCDHMM_Results: test6
                Enter the number of cases (at most 5):
                4
                Enter the M value for case 1 : 80
                Enter the N values for classes (in the order ['a', 'ai', 'chA', 'dA', 'lA']) for case 1 separated by spaces : 5 5 5 5 5
                Enter the M value for case 2 : 50
                Enter the N values for classes (in the order ['a', 'ai', 'chA', 'dA', 'lA']) for case 2 separated by spaces : 4 4 4 4 4
                Enter the M value for case 3 : 60 
                Enter the N values for classes (in the order ['a', 'ai', 'chA', 'dA', 'lA']) for case 3 separated by spaces : 3 4 4 4 5
                Enter the M value for case 4 : 100
                Enter the N values for classes (in the order ['a', 'ai', 'chA', 'dA', 'lA']) for case 4 separated by spaces : 4 4 4 5 5 
                .....
            '''
                where ..... is output from train_hmm and test_hmm execution

            Result:-

                A folder 'test6_HCDHMM_Results' will be generated and it will have the confusion matrices, a comparative ROC plot and 
                a comparative DET plot.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

***) Optional Part HMM on connected spoken Digits(CSD data):- (the script is in chmm_CSD.py)

        Place the folder F containing the CSD data in the code directory. F must contain a subdirectory dev/ which has the development data and another subdirectory
        test/ which has the blind test data as per the formats in the assignment drive folders. 
        Also, the codebook for quantisation and the model files for ISD HMMs, for all classes - 1, 4, 6, o, z, must also be placed in the code directory.
        All these files are generated by hmm_ISD.py script. The hmm executables (train_hmm and test_hmm) must also be there in the code directory.

        Actual Input and Output:- (assuming F = CSD and the codebook file is kmeans.codebook)

        i) for cmd $ python3 chmm_ISD.py > optional.out
           
            Input:-
            '''
                CISD
                1 4 6 o z
                80
                kmeans.codebook
            '''

            Output will be in optional.out file from which we get the accuracy on dev data and the predictions for blind data as

            '''
            .....
                Accuracy = 0.29545454545454547
                The predictions on blind test data are: 
                3.mfcc -> oo4
                1.mfcc -> z66
                2.mfcc -> z44
                5.mfcc -> z46
                4.mfcc -> zz6
            '''
            where ..... indicates train_hmm and test_hmm output
----------------------------------------------------------------------------------------------------------------------------------------------------------------
