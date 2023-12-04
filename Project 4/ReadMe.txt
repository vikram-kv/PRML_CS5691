TEAM - 06
COURSE - CS5691 (Pattern Recognition and Machine Learning)
MEMBERS - K V Vikram (CS19B021) and Vedant Saboo (CS19B074)

----------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------


PART A - KNN :-


    The code files are knn.py and knn_pcaandlda.py. The details are

        i) knn.py -> python3 knn.py -t {none,pca,lda} -d {synthetic,image,spoken_digit,handwritten} [-n {1,2,3,4,5}]

            -t option is for transformation choice

            -d option is for type of data

            -n option is for number of cases to consider and if n > 1, a comparative ROC/DET plot is generated. This option is not required
            with default value = 1.

            Suppose we have spoken digit data in ISD.

            Sample I/O for "$ python3 knn.py -t lda -d spoken_digit -n 5 " :-

                Enter the name of the folder containing the isolated digits data : ISD
                Enter the dimension required after LDA: 4
                Enter the k values separated by spaces: 5 20 40 60 80
                k = 5 : 81.67
                k = 20 : 81.67
                k = 40 : 81.67
                k = 60 : 80.00
                k = 80 : 75.00

            We note that the accuracy for each value of k is printed. Also a folder "plots_KNN_ISD_LDATRANSFORM" is created with 
            confusion matrices for each case and a comparative ROC/DET plot. Similarly well-named folders with plots are generated for
            other data sets and transformations as well.

            Also, python3 knn.py --help will display a useful help message.


       ii) knn_pcaandlda.py-> python3 knn_pcaandlda.py -d {synthetic,image,spoken_digit,handwritten} [-n {1,2,3,4,5}]

            -d option is for type of data

            -n option is for number of cases to consider and if n > 1, a comparative ROC/DET plot is generated. This option is not required
            with default value = 1.

            This code will perform PCA followed by LDA on the given data and hence, there is no transformation option like the previous file.

            Suppose we have spoken digit data in ISD.

            Sample I/O for "$ ppython3 knn_pcaandlda.py -d spoken_digit -n 5" :-

                Enter the name of the folder containing the isolated digits data : ISD
                Enter the dimension required after PCA: 50
                Enter the dimension required after LDA: 4
                Enter the k values separated by spaces: 5 20 40 60 80
                k = 5 : 93.33
                k = 20 : 93.33
                k = 40 : 93.33
                k = 60 : 95.00
                k = 80 : 95.00

            We note that the accuracy for each value of k is printed. Also a folder "plots_KNN_ISD_PCATRANSFORM_LDATRANSFORM" is created with 
            confusion matrices for each case and a comparative ROC/DET plot. Similarly well-named folders with plots are generated for
            other data sets as well.

            Also, python3 knn_pcaandlda.py --help will display a useful help message.


----------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------


PART B - SVM :-


    The code files are svm.py and svm_pcaandlda.py. The details are

        i) svm.py - this corresponds to knn.py. The usage is exactly the same. Here, the user needs to give choices for kernel types
                    rather than 'k' values of KNN. Allowed kernels are rbf, linear, poly and sigmoid. For poly kernel, the user can
                    choose the degree as well. For each kernel, the accuracy is printed and a comparative ROC/DET curve along with the
                    confusion matrices are saved in a well-named folder.


       ii) svm_pcaandlda.py-> this corresponds to knn_pcaandlda.py. The usage is exactly the same. Here, the user needs to give choices for kernel types
                    rather than 'k' values of KNN. Allowed kernels are rbf, linear, poly and sigmoid. For poly kernel, the user can
                    choose the degree as well. For each kernel, the accuracy is printed and a comparative ROC/DET curve along with the
                    confusion matrices are saved in a well-named folder. We note again that PCA followed by LDA is done here.


----------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------


PART C - Logistic Regression :-




----------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------


PART D - ANN :-




----------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------


HELPER FILES(NOT RUNNABLE):-

      i) transformmodule.py -> Contains code for PCA dimension reduction and LDA transformation.

     ii) plotmmodule.py -> Contains helper functions for plotting ROC/DET curves and confusion matrices.

    iii) inputmodule.py -> Contains code for getting input from files of different datasets, pre-processing(normalizing them) the data if needed,
                           and performing the rolling window average transformation on variable length series to convert the vectors to a fixed 
                           length. We perform this transformation on spoken digits data and handwritten character data.
     
     iv) knnmodule.py -> Contains code for the class KNN with associated functions which is used in knn.py and knn_pcaandlda.py 