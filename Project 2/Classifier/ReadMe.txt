Instructions for running:-

      i) The code for this problem is split into multiple modules to promote readability.
     ii) The command for execution is "python3 classifier.py" after opening this folder on terminal.
    iii) The text files containing the train data(say train.txt) and the development data(say dev.txt)
         should be placed in this folder.Then, you can give 'train.txt' when prompted for 
         training data file and 'dev.txt' when prompted for development data.
     iv) You will then be asked for a name. If we give X as input, then a folder X_plots will be created in 
         the working directory.
      v) The code will generate all the relevant plots for each of the 5 cases(as in the problem statement) + the comparative
         ROC-DET curve within X_plots folder. Each plot will also be displayed in a new window.
     vi) Each plot will have a suffix "_case{N}.png" where N indicates its case number between 1 and 5.
    vii) We note that orientation for the PDF plot may not be properly captured in the saved file. However, we can re-orient the
         displayed PDF plot to get a good view.

Demo:-

    Say my training data and development data is in this directory as 'train.txt' and 'dev.txt' respectively. I want my plots
    inside 'RealData_plots' folder in this directory.
    Now, the terminal I/O should be as below.

        Enter file name for training data with rel. path:train.txt
        Enter file name for development data with rel. path:dev.txt
        Enter a name nm (All plots will be put in a new folder (nm)_plots):RealData

    This results in the below structure for the RealData_plots folder.

        RealData_plots
        ├── ConfusionMatrix_Case1.png
        ├── ConfusionMatrix_Case2.png
        ├── ConfusionMatrix_Case3.png
        ├── ConfusionMatrix_Case4.png
        ├── ConfusionMatrix_Case5.png
        ├── DecisionBoundary_Case1.png
        ├── DecisionBoundary_Case2.png
        ├── DecisionBoundary_Case3.png
        ├── DecisionBoundary_Case4.png
        ├── DecisionBoundary_Case5.png
        ├── EigenvectorContour_Case1.png
        ├── EigenvectorContour_Case2.png
        ├── EigenvectorContour_Case3.png
        ├── EigenvectorContour_Case4.png
        ├── EigenvectorContour_Case5.png
        ├── PDFs_Case1.png
        ├── PDFs_Case2.png
        ├── PDFs_Case3.png
        ├── PDFs_Case4.png
        ├── PDFs_Case5.png
        └── ROCDETPlot.png
