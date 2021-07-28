Required Python Version: v3.8.2

Make sure that pandas, numpy, matplotlib and seaborn libraries are installed.
If they are not installed, execute below commands in terminal:
    $ pip3 install pandas
    $ pip3 install numpy
    $ pip3 install matplotlib
    $ pip3 install seaborn

After making sure all libraries necessary to run the program in your computer,
simply call the below command in terminal within the directory where the
q3main.py file is located
    $ python3 q3main.py

After executing the program, you will first see the training data statistics,
including total number of emails in the training set, number of spam emails in
training set and number of normal emails in training set.

Then, the program will use three Naive Bayes Models to predict the emails being
spam or normal, in the order of Multinomial Naive Bayes Model Without Smoothing,
Multinomial Naive Bayes Model With Smoothing and Bernoulli Naive Bayes Model.

For each model, you will see the total number of predictions, number of true
predictions, number of wrong predictions, and the rate for accuracy for each
model respectively, in the terminal.

IMPORTANT NOTE: For each model, confusion matrix is shown as in a plot form. You need
to first close the corresponding plot to the current model to make the program
continue executing.