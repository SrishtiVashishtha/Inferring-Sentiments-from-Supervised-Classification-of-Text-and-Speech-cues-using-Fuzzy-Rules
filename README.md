# Inferring-Sentiments-from-Supervised-Classification-of-Text-and-Speech-cues-using-Fuzzy-Rules
Multimodal Sentiment Analysis of video reviews on social media platform, using a supervised fuzzy rule-based system.

**Code for the paper**

[Inferring Sentiments from Supervised Classification of Text and Speech cues using Fuzzy Rules](https://www.sciencedirect.com/science/article/pii/S1877050920308140?via%3Dihub)


**Description**

This paper introduces a supervised fuzzy rule-based system for multimodal sentiment classification, which can
identify the sentiment expressed in video reviews on social media platform. It has been demonstrated that multimodal sentiment
analysis can be effectively performed by the joint use of linguistic and acoustic modalities. In this paper computation of the
sentiment using a novel set of fuzzy rules has been done to classify the review into: positive or negative sentiment class. The
confidence score from supervised Support Vector Machine (SVM) classification of text and speech cues is considered as the input
variable for the fuzzy rules. 

**Dataset**
We have used [CMU-MOSI](https://arxiv.org/ftp/arxiv/papers/1606/1606.06259.pdf) Please cite the creators of this dataset. This Dataset can be downloaded from [here](http://immortal.multicomp.cs.cmu.edu/raw_datasets/). 

**Running the model:**

_TextFeatures.py_: the code for implementing unimodal textfeatures based SVM and computation of Text Confidence Scores.

_SpeechFeatures.py_ : the code for implementing unimodal textfeatures based SVM and computation of Speech Confidence Scores.

_FuzzyRulebasedSupervisedClassifier.py_ :the code for implementing the fuzzy rule based system using Text and Speech Confidence Scores.

**Citation**

If using this code, please cite our work using :

>Vashishtha, Srishti, and Seba Susan. "Inferring Sentiments from Supervised Classification of Text and Speech cues using Fuzzy Rules." Procedia Computer Science, 167 (2020), pp.1370-1379.
