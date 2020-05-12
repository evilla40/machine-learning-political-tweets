#%%
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
#%%
#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
     #Remove appropiate apostrophes ('s) from string
    tempText = ''
    apos = 0
    for t in text:
        if (t == '\'' or t == '’'):
            apos = 1
        else:
            if (apos == 1 and t == 's'):
                apos = 0
                continue
            else:
                apos = 0
            tempText += t
            
    #Remove valid urls        
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    all_urls = re.findall(pattern, tempText)
    for url in all_urls:
        if (url[len(url) - 1] == ')'):
            url = url[:-1]
        newText = ''
        if(re.search(url, tempText)):
            begin, end = re.search(url, tempText).span()
            for i in range(len(tempText)):
                if (i >= begin and i < end): continue
                else: newText += tempText[i]
            tempText = newText
            
    #Break words at punctuations/remove punctuations
    finalText = ''
    for char in tempText:
        if (char in string.punctuation or char == '—' or char == '–'):
            char = ' '
        finalText += char
    
    #Tokenize string
    final_list = nltk.word_tokenize(finalText.lower())

    #Pos tagging and lemmatzing
    pos = nltk.pos_tag(final_list)
    return_list = []
    for t in pos:
        if(t[1][0] == 'V'): p = 'v'
        elif(t[1][0] == 'J'): p = 'a'
        elif(t[1][0] == 'R'): p = 'r'
        else: p = 'n'
        temp = lemmatizer.lemmatize(t[0], p)
        return_list.append(temp)
    
    return return_list
    
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    df["text"] = df["text"].apply(lambda row: process(row))
    return df
    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    vector = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stop_words, min_df=2)
    array = []
    for t in processed_tweets["text"]:
        string = ''
        for word in t:
            string += word
            string += ' '
        array.append(string)
    sparse_matrix = vector.fit_transform(array)
    return vector, sparse_matrix

#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    array = np.zeros(len(processed_tweets["screen_name"]))
    curr = processed_tweets["screen_name"].apply(lambda row: process(row))
    counter = 0
    for c in curr:
        if(c[0] == "gop" or c[0] == "mike" or c[0] == "realdonaldtrump"): array[counter] = 0
        else: array[counter] = 1
        counter += 1
    return array

#%%
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize
        """
        self.mode = None
        
    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        """
        count0 = 0
        count1 = 0
        for i in y:
            if(i == 0): count0 += 1
            else: count1 += 1
        if(count1 > count0):
            self.mode = 1
        else: self.mode = 0
        return self
    
    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        array = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            array[i] = self.mode
        return array

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.classes.SVC: classifier learnt from data
    """  
    svc = sklearn.svm.classes.SVC(kernel=kernel)
    svc.fit(X_train, y_train)
    return svc

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.classes.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    acc = sklearn.metrics.accuracy_score(classifier.predict(X_validation), y_validation)
    return acc

#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.classes.SVC: classifier learnt
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    proc_tweets = process_all(unlabeled_tweets)
    array = []
    for t in proc_tweets["text"]:
        string = ''
        for word in t:
            string += word
            string += ' '
        array.append(string)
    sparse_matrix = tfidf.transform(array)
    val = classifier.predict(sparse_matrix)
    return val