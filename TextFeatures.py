import time
start=time.time()

import pickle
import pandas as pd
import math
from nltk.tokenize import word_tokenize

# Loading of Text Transcripts of Dataset
df=pd.read_csv("C:/MyData/PythonPractice/Speech_CMU-MOSI/original_transcripts.csv",encoding='ISO-8859-1')

#print(df.columns.tolist())
#filename=df[['FileName']] # extracting only the filenames
#fileID=df[['FileID']] # extracting only the fileids
#utterance=df[['Utterance']]  # extracting only utterances
#utterancelist=[]
#for row in utterance.iterrows():
#    index, data = row
#    utterancelist.append(data.tolist()[0])  # data.tolist() gives value [[1],[0],...] so we just need the label
#
##print(utterancelist)
#print(len(utterancelist)) #2199
#
#filename = 'utterances.pickle'
#outfile = open(filename,'wb')
#pickle.dump(utterancelist,outfile)
#outfile.close()

#seq=df[['SeqNo']]  # extracting only the sequence numbers
#label=df[['Label']] # extracting only the sentiment labels
#
##print(type(label)) #dataframe
#
#print(len(filename))   #Total no of utterances in dataset
#N=len(filename)
#
#seqlist=list(set(df.SeqNo))  # sequence of series type
#print(seqlist)        #Unique set of sequence numbers
#
# 
##sentiment_labels=[] # converts rows of data frame to list
##
##for row in label.iterrows():
##    index, data = row
##    sentiment_labels.append(data.tolist()[0])  # data.tolist() gives value [[1],[0],...] so we just need the label
##
##print(sentiment_labels)
###print(len(temp)) #2199
#
#u=df.Utterance
#utterances=[]
#for x in u:
#    utterances.append(x)
#print(utterances)    #Unique set of file names
#print(len(utterances))
#
## BAG OF WORDS
#global_unique_tokens=[]
#
##for i in range(0,2):
#for i in range(len(utterances)):
#        doc=utterances[i]
#        allwords = word_tokenize(doc)
#        allwords=[word.lower() for word in allwords if word.isalpha()]
##        print(allwords)
#        unique_tokens = []
#        for x in allwords:
#            if x not in unique_tokens:
#                unique_tokens.append(x)
#                global_unique_tokens.append(x)  
##        print(unique_tokens)
##print(global_unique_tokens)
#
#BOW=[]
#for x in global_unique_tokens:
#        if x not in BOW:
#            BOW.append(x)
#print(" Bag of Words: ")
#print(BOW)
#print(len(BOW))

## CALCULATION OF IDF VALUES 
#
## IDF= log(N/n) N : NUMBER OF DOCUMENTS/ Sentences  n : NUMBER OF DOCUMENTS A TERM HAS APPEARED IN
#
## n : NUMBER OF DOCUMENTS A TERM HAS APPEARED IN
##
##count=[]
##for k in range(len(BOW)):
##    x=0
##    for i in range(len(utterances)):
##
##            doc=utterances[i]
##            allwords = word_tokenize(doc)
##            allwords=[word.lower() for word in allwords if word.isalpha()]
###            print(allwords)
##           
##            if BOW[k] in allwords:
##                x=x+1
##    count.append(x)
##print(count) 
##print(len(count))
##ndict=dict(zip(BOW,count))
##print(ndict)
##
##IDF=[]
##for i in range(len(BOW)):
##    x=math.log10(N/(count[i]))
##    x=round(x,3)
##    IDF.append(x)
##print("\n Pair wise --(Words,IDF Values):" )
##print(IDF)
##print(len(IDF))
##IDFdict=dict(zip(BOW,IDF))
##print(IDFdict)
#
#
##filename = 'simple_IDF.pickle'
##outfile = open(filename,'wb')
##pickle.dump(IDFdict,outfile)
##outfile.close()
#

#filename='C:/Users/srish/Dropbox/DTU/Research/4 May 19/simple_IDF.pickle'
#infile = open(filename,'rb')
#IDFdict = pickle.load(infile, encoding='latin1')
#infile.close()
#print(IDFdict)


#global_TF_Matrix=[]
#global_TF_IDF_Matrix=[]
#
#
#for i in range(len(utterances)):
##for i in range(0,2):
##        print(utterances[i])
#    doc=utterances[i]
##    print(doc)
#    allwords = word_tokenize(doc)
#    allwords=[word.lower() for word in allwords if word.isalpha()]
##    print(allwords)
#    
#    unique_tokens = []
#    for x in allwords:
#        if x not in unique_tokens:
#            unique_tokens.append(x)
#    
#    wordfreq = [] #   NUMBER OF TIMES A TERM APPEAR IN EACH SENTENCE
#    # len(uniquetokens) #   NUMBER OF TERMS IN  EACH REVIEW
#   
#    for w in allwords:
#        wordfreq.append(allwords.count(w))
##    print("\n Pair wise --(Words,Frequences):" )
##    print(list(zip(allwords, wordfreq)))
##    print("\n Number of times a term appear :")
##    print(wordfreq)
#    
#    # TF for each term
#    TF=[]
#    for j in range(len(allwords)):
#        x=((wordfreq[j])/(len(unique_tokens)))
#        x=round(x,3)
#        TF.append(x)
##    print("\n TF values : ")
##    print(TF)
#        
##    print("\n Pair wise --(Words,TF Values):" )
#    TFdict=dict(zip(allwords, TF))
##    print(TFdict)
#    
#    TF_Matrix=[]
#    TFrow=[]
#    for k in range(len(BOW)):
#        if BOW[k] in allwords:  
#            TFrow.append(TFdict[BOW[k]])
#        else:
#            TFrow.append(0)
##    print(TFrow)
#    TF_Matrix.append(TFrow)
#    
#    TF_IDF_Matrix=[]
#    TF_IDF_row=[]
#    for k in range(len(BOW)):
#        if BOW[k] in allwords:
#            t=TFdict[BOW[k]]*IDFdict[BOW[k]]
#            TF_IDF_row.append(round(t,3))
#        else:
#            TF_IDF_row.append(0)
##    print(TF_IDF_row)
#    TF_IDF_Matrix.append(TF_IDF_row)
#
##        print("\n TF MATRIX :" +str(TF_Matrix))
##        print(len(TF_Matrix))
##        print("\n TF-IDF MATRIX: " +str(TF_IDF_Matrix))
##        print(len(TF_IDF_Matrix))
#
#    global_TF_Matrix.append(TFrow)
#    global_TF_IDF_Matrix.append(TF_IDF_row)
#        
##print("\n Global TF MATRIX :" +str(global_TF_Matrix))
#print(len(global_TF_Matrix))
##print("\n Global TF-IDF MATRIX :" +str(global_TF_IDF_Matrix))
#print(len(global_TF_IDF_Matrix))
#
#textfeatures=global_TF_IDF_Matrix
#import pickle
#filename = 'textfeatures_TF-IDF_normalize.pickle'
#outfile = open(filename,'wb')
#pickle.dump(global_TF_IDF_Matrix,outfile)
#outfile.close()


###### COLUMN WISE ( FEATURE WISE) NORMALIZATION
#filename='C:/Users/srish/Dropbox/DTU/Research/4 May 19/textfeatures_TF-IDF_simple.pickle'
#infile = open(filename,'rb')
#textfeatures = pickle.load(infile, encoding='latin1')
#infile.close()
#print(textfeatures)
#
#rowsize=len(textfeatures)
##print(rowsize)  #2199
#colsize=len(textfeatures[0])
##print(colsize)  #3079
#
##Transpose the TextFeatures Matrix for columnwise normalization
#newtextfeatures = [[textfeatures[j][i] for j in range(len(textfeatures))] for i in range(len(textfeatures[0]))] 
#
#newrowsize=len(newtextfeatures)
##print(newrowsize)  #3079
#newcolsize=len(newtextfeatures[0])
##print(newcolsize)  #2199
#
##import csv
#
#datacol1=newtextfeatures[0]   # 1st BOW feature list
#norm = [(round((i- min(datacol1))/(max(datacol1)-min(datacol1)),3)) for i in datacol1]
#print(norm)
#
#normalized_data=[]
#
#for j in range(colsize):
#    datacol1=newtextfeatures[j]   # 1st BOW feature list
#    norm = [(round((i- min(datacol1))/(max(datacol1)-min(datacol1)),3)) for i in datacol1]
##    print(norm)
#    normalized_data.append(norm)
#
##Transpose it back
#finalnormalizedfeatures = [[normalized_data[j][i] for j in range(len(normalized_data))] for i in range(len(normalized_data[0]))] 
#
#filename = 'textfeatures_normalized.pickle'    #Save Text features in pickle file
#outfile = open(filename,'wb')
#pickle.dump(finalnormalizedfeatures,outfile)
#outfile.close()

filename='C:/Users/..../textfeatures_normalized.pickle'  # Path of Textfeatures pickle file
infile = open(filename,'rb')
textfeatures = pickle.load(infile, encoding='latin1')
infile.close()
#print(textfeatures)

filename='C:/Users/.../sentimentlabels_simple.pickle'    #Path of Sentiment Label pickle file
infile = open(filename,'rb')
sentiment_labels = pickle.load(infile, encoding='latin1')
infile.close()
#print(sentiment_labels)
#
################# SVM CLASSIFIER ###################

### Import train_test_split function
from sklearn.model_selection import train_test_split

accuracy=[]
precision=[]
recall=[]
fscore1=[]
fscore2=[]

t=10  # Number of folds for:- Cross Validation
for i in range(0,t):
    
    print(i+1)     
    # Split dataset into training set and test set     # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(textfeatures, sentiment_labels, test_size=0.3)
       
    #Import svm model
    from sklearn import svm
      
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel
    
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    
#    #decision function
    decision=clf.decision_function(X_test)   # size= size of textfeatures
    
    filename = 'TextConfidenceScore.pickle'    # Text Confidence Score
    outfile = open(filename,'wb')
    pickle.dump(decision,outfile)
    outfile.close()
    
    filename = 'Text_TrueLabels.pickle'
    outfile = open(filename,'wb')
    pickle.dump(y_test,outfile)
    outfile.close() 
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    filename = 'Text_PredictedLabels.pickle'
    outfile = open(filename,'wb')
    pickle.dump(y_pred,outfile)
    outfile.close()
    
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    
    # Model Accuracy: how often is the classifier correct?
    acc=metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:",acc)
    
    # Model Precision: what percentage of positive tuples are labeled as such?
    prec=metrics.precision_score(y_test, y_pred, average='macro')
    print("Precision:",prec)
    
    # Model Recall: what percentage of positive tuples are labelled as such?
    re=metrics.recall_score(y_test, y_pred, average='macro')
    print("Recall:", re)
    
    # Model F1- Score: 
    f1=metrics.f1_score(y_test, y_pred, average='macro')
    print("F1-Score:",f1)
      
    accuracy.append(acc)
    precision.append(prec)
    recall.append(re)
    fscore1.append(f1)
 
avg_accuracy=sum(accuracy)/t
print(" Avg Accuracy:",avg_accuracy*100)
avg_precision=round(sum(precision)/t,3)
print(" Avg Precision:",avg_precision)
avg_recall=round(sum(recall)/t,3)
print(" Avg Recall:",avg_recall)
avg_fscore1=round(sum(fscore1)/t,3)    
print(" Avg F1-score :",avg_fscore1)

end=time.time()
print(str(end-start)+" secs")