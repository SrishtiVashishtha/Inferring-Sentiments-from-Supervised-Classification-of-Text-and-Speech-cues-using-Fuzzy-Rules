import time
start=time.time()
import pickle 

filename='C:/Users/..../IS13_normalize.pickle'  # Path of Speechfeatures pickle file
infile = open(filename,'rb')
IS13features = pickle.load(infile, encoding='latin1')
infile.close()
#print(IS13features)

filename='C:/Users/...../sentimentlabels_simple.pickle'   # Path of Sentiment Label pickle file
infile = open(filename,'rb')
sentiment_labels = pickle.load(infile, encoding='latin1')
infile.close()
#print(sentiment_labels)

############### SVM CLASSIFIER ###################

# Split dataset into training set and test set

# Import train_test_split function
from sklearn.model_selection import train_test_split

accuracy=[]
precision=[]
recall=[]
fscore1=[]
fscore2=[]

t=10  # Number of folds for:- Cross Validation
for i in range(0,t):
    
    print(i+1)     
    # Split dataset into training set and test sets    # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(IS13features, sentiment_labels, test_size=0.3)
    
    #Import svm model
    from sklearn import svm
      
    #Create a svm Classifier
    clf = svm.SVC(C=10,kernel='rbf',gamma='auto') # Rbf Kernel
    
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    
#    #decision function
    decision=clf.decision_function(X_test)   # size= size of audiofeatures
    
    filename = 'SpeechConfidenceScore.pickle'   # Speech Confidence Score
    outfile = open(filename,'wb')
    pickle.dump(decision,outfile)
    outfile.close()
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
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
 
avg_accuracy=round(sum(accuracy)/t,3)
print(" Avg Accuracy:",avg_accuracy*100)
avg_precision=round(sum(precision)/t,3)
print(" Avg Precision:",avg_precision)
avg_recall=round(sum(recall)/t,3)
print(" Avg Recall:",avg_recall)
avg_fscore1=round(sum(fscore1)/t,3)    
print(" Avg F1-score:",avg_fscore1)

end=time.time()
print(str(end-start)+" secs")