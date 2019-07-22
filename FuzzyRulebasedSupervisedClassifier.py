import numpy as np
import skfuzzy as fuzz
#from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

import time
start = time.time()
import pickle 

filename='C:/Users/..../TextConfidenceScore.pickle'   # Path of Text Confidence Score
infile = open(filename,'rb')
decision_text = pickle.load(infile, encoding='latin1')
infile.close()
#print(decision_text)

#normalization of scores

norm_decision_text = [(round((i- min(decision_text))/(max(decision_text)-min(decision_text)),3)) for i in decision_text]
#print(norm_decision_text)

filename='C:/Users/..../SpeechConfidenceScore.pickle'  # Path of Speech Confidence Score
infile = open(filename,'rb')
decision_audio = pickle.load(infile, encoding='latin1')
infile.close()
#print(decision_audio)

#normalization of scores

norm_decision_audio = [(round((i- min(decision_audio))/(max(decision_audio)-min(decision_audio)),3)) for i in decision_audio]
#print(norm_decision_audio)

filename='C:/Users/.../Text_TrueLabels.pickle'   # Path of Text TrueLabels
infile = open(filename,'rb')
true_labels = pickle.load(infile, encoding='latin1')
infile.close()
#print(true_labels)

filename='C:/Users/.../Text_PredictedLabels.pickle' #Path of Text Predicted Labels
infile = open(filename,'rb')
predicted_labels = pickle.load(infile, encoding='latin1')
infile.close()
#print(predicted_labels)

# Generate universe variables
#   * pos and neg on subjective ranges [0, 7]
#   * op has a range of [0, 11] in units of percentage points
x_t = np.arange(0, 1, 0.1)
x_a = np.arange(0, 1, 0.1)

#       FUZZIFICATION  UISNG TRIANGULAR FUZZY MEMBERSHIP
  
# Generate fuzzy membership functions
t_lo = fuzz.trimf(x_t, [0, 0, 0.5])
t_hi = fuzz.trimf(x_t, [0.4, 1, 1])
a_lo = fuzz.trimf(x_a, [0, 0, 0.5])
a_hi = fuzz.trimf(x_a, [0.4, 1, 1])

## Visualize these universes and membership functions
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 6))
#
ax0.plot(x_t, t_lo, 'b', linewidth=1.5, label='Low')
ax0.plot(x_t, t_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('Text')
ax0.legend()

ax1.plot(x_a, a_lo, 'b', linewidth=1.5, label='Low')
ax1.plot(x_a, a_hi, 'r', linewidth=1.5, label='High')
ax1.set_title('Speech')
ax1.legend()

## Turn off top/right axes
#for ax in (ax0, ax1):
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.get_xaxis().tick_bottom()
#    ax.get_yaxis().tick_left()
#
#plt.tight_layout()

senti=[]

t=len(true_labels)
t=10  #  10- FOLD CROSS VALIDATION

for j in range(0,t):
    print(j)
#    print("Utterance:"+str(j+1)+" "+str(reviews[j]))
    text=norm_decision_text[j]      
    audio=norm_decision_audio[j]
    
    print("\nConfidence score of Text: "+str(text))    
    print("Confidence score of Audio: "+str(audio))
       
    # Normalized confidence scores are  fuzzified using triangular membership function. 
    # We need the activation of our fuzzy membership functions at these values.
    t_level_lo = fuzz.interp_membership(x_t, t_lo, text)
    t_level_hi = fuzz.interp_membership(x_t, t_hi, text)   
    
    a_level_lo = fuzz.interp_membership(x_a, a_lo, audio)
    a_level_hi = fuzz.interp_membership(x_a, a_hi, audio)
    
    # FORMULATION OF RULES
    
    # Now we take our rules and apply them. Rule 1 concerns bad food OR nice.
    # The OR operator means we take the maximum of these two.
    active_rule1 = np.fmin(t_level_lo, a_level_lo)
    active_rule2 = np.fmin(t_level_hi, a_level_lo)
    active_rule3 = np.fmin(t_level_lo, a_level_hi)
    active_rule4 = np.fmin(t_level_hi, a_level_hi)

    print("Firing Strength of Rules:")
    print("wr1: "+str(active_rule1))
    print("wr2: "+str(active_rule2))
    print("wr3: "+str(active_rule3))
    print("wr4: "+str(active_rule4))

    print("\nPredicted Label by unimodal SVM: "+str(predicted_labels[j]))
    print("True Label: "+str(true_labels[j]))
    
    if(predicted_labels[j]==true_labels[j]):
        senti.append(predicted_labels[j])
        
        print("Fuzzy Output(z): "+str(predicted_labels[j]))
        if(predicted_labels[j]==0):
            print("\nOutput after Defuzzification: Negative")
        else:
            print("\nOutput after Defuzzification: Positive")  
    else:
        # Rule 1 Rule 3 : FLIP
        # Rule 2 Rule 4: RETAIN
        
        label_flip= 1-predicted_labels[j]
        label_retain=predicted_labels[j]
        
    #     DEFUZZIFICATION
            
        sumwz=(active_rule1*label_flip+active_rule2*label_retain+active_rule3*label_flip+active_rule4*label_retain)
        sumw=(active_rule1+active_rule2+active_rule3+active_rule4)
        output=sumwz/sumw
        
        print("Fuzzy Output(z): "+str(output))
                   
        if 0<=(output)<0.5:    # R
            print("\nOutput after Defuzzification: Negative")
    #        print("Negative \n")
            senti.append(0)
        elif 0.5<=(output)<=1:
            print("\nOutput after Defuzzification: Positive")
    #        print(" Positive \n ")
            senti.append(1)
       
    print("Review actual sentiment: "+ str(true_labels[j]))    
    print("(0: Negative, 1: Positive)\n")

count=0
for k in range(0,t):
    if(true_labels[k]==senti[k]):
        count=count+1
#print(count)

print("Accuracy is: "+ str(round((count/t*100),3)))

from sklearn import metrics
y_true = true_labels
y_pred = senti

acc=metrics.accuracy_score(y_true, y_pred)
print("Accuracy score:",str(round(acc*100,3)))

p1=metrics.precision_score(y_true, y_pred, average='macro')  
p2=metrics.precision_score(y_true, y_pred, average='micro')  

print("Precision score: " + str(round(p1,3)))

r1=metrics.recall_score(y_true, y_pred, average='macro')  

print("Recall score: " + str(round(r1,3)))

f1=metrics.f1_score(y_true, y_pred, average='macro')  

print("F1 score: " + str(round(f1,3)))

end=time.time()
print(str(end-start)+" secs")