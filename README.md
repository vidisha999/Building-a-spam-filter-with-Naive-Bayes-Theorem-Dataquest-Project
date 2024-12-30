# Building-a-spam-filter-with-Naive-Bayes-Theorem-Dataquest-Project

## Project Description

The goal of this project tio build a spam filter to classify messages using the multinomial Naive Bayes algorithm, which  estimates the  probabilities and classifies new messages by calculating the probabillities for both spam and non spam messages.
The classification made by Naive Bayes algorithm is based on the probability values.If the probability of the spam messages is higher than that of the non-spam messages, it classifies the new message as a spam message.

## Project Background 

Multinomial Naive Bayes (MNB) is a probabilistic classifier  that is based on Bayes' theorem, which is commonly used  to calculate the probability distribution of text data, with features that represent discrete frequencies or counts of events in various natural language processing (NLP) tasks. 

The Bayesian model is a supesrvised learning model that estimates the posterior probability of a event given the prior probability of the model, once more evidence becomes available. The Naive Bayes theorem uses the basic assumption, features in a dataset are independent and there are three main types of it, which include Gaussian Naive Bayes for continuous data, Multinomial Naive Bayes for discrete data, and Bernoulli Naive Bayes for binary/boolean features.

 ## Data 
 This classification task is performed on a dataset of 5,572 messages extracted from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

 ## Exploratory Data Analysis

 The follwoing insights were gained by exploring the dataset: 
  - The dataset contains 86.6% of non-spam messages and 13.5% of spam messages.
  - The dataset was randomized before splitting into training and testing data, so that the both splits contain the similar percentages of spam and non-spam messages.

## Creating Vocabularoy 

A vocabulary, which is a list containing all unique words in the dataset is created using the following code : 
```python
training_set['SMS']=training_set['SMS'].str.split()
vocabulary=[]
for sms in enumerate(training_set['SMS']):
     for word in sms:
          vocabulary.append(sms)
vocabulary=list(set(vocabulary))
```
          
## Creating Dictionary with Unique Word count 

- As the first step, the dictionary created with count of the unique words in the vocabulary is transformed to a dataframe.
- Then this dataframe was concated with the the original training datset to create a new dataset, displaying the SMS message and count of each word in that message.
- The following code is used to create the dictionary of unique word count:

  ``` python
  dictionary={unique_word: [0]* len(training_set['SMS']) for unique_word in vocabulary}
  for index,sms in training_set['SMS']:
        for word in sms:
             dictionary[word][index]+=1
  ```

 ## Applying the Naive Bayes Theorem



1. The posterior probability of a message being spam given the words in the message can be represented as:

   **```P(Spam|w1,w2,...,wn) ∝ P(Spam)⋅ n∏i=1 P(wi|Spam)```**

3.  The posterior probability of a message being non-spam given the words in the message can be represented as:

    **```P(Ham|w1,w2,...,wn) ∝ P(Ham)⋅ n∏i=1 P(wi|Ham)```**
4. The probability of a word wi given that the message is spam can be calculated as:

     **```P(wi|Spam)= {(N_wi|Spam) + α} /(N_Spam+α⋅N_Vocabulary)```**
5.  The probability of a word wi given that the message is non-spam can be calculated as:

     **```P(wi|Ham)={(N_wi|Ham) + α} /(N_Ham+α⋅N_Vocabulary)```**


Where:

- **```P(Spam | w1, w2, ..., wn)```** : Probability that the message is spam given the words.
- **```P(Spam)```**  : Prior probability of spam.
- **```P(wi | Spam)```** : Probability of each word wi occurring in spam messages.
-  **```P(Ham | w1, w2, ..., wn)```** : Probability that the message is non-spam given the words.
- **```P(Ham)```**  : Prior probability of non- spam.
- **```P(wi | Ham)```** : Probability of each word wi occurring in non-spam messages.
- **```Π```**   : Product over all words in the message.
- **```N_wi|Spam```**    : The number of times the word wi appears in spam messages.
- **```N_Spam```**     : The total number of words in spam messages.
 - **```N_wi|Ham```**    : The number of times the word wi appears in non-spam messages.
- **```N_Ham```**     : The total number of words in non-spam messages.
- **```α```**         : Smoothing parameter (Laplace smoothing).
- **```N_Vocabulary```** : The total number of unique words in the vocabulary.

- Laplace Smoothing (**α**=1) was used to avoid zero probabilities of the words that may not appear in the spam or non-spam messages.

1. All above parameters are calculated as shown in the [Notebook](Notebook/vidisha_Using+Naive+Bayes+to+build+a+spam+filter.ipynb)
2. By using above parameters, both probabilities of the new message being spam message or a non-spam message was calculated.
3. Depending on highest proabability out of them, the new message was classified as a spam or non-spam message.

## Determing the accuracy of the predicted labels 

The accuracy of the predicted labels can be calculated by comparing the original label of the messages in the test  dataset to the predicted label. 
The following code can be used to calculate the accuracy of the prediction: 
Furthermore the SMS message of incorrectly predicted labels can be collected into a list and analyze the each  word of it to understand what most probably caused them to get classified incorrectly.

```python
test_set['predicted'] = test_set['SMS'].apply(classify_test_set) # classify_test_set function is defined in the Notebook
total=test_set.shape[0]

for row in test_set.iterrows():
    row=iterrows[1]
    if row['Label']==row['predicted']:
         correct=+1
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)
```

## Conclusion 

An accuracy of 98% suggests that the Naive bayes Classifier is reliable for spam detection in a given dataset of text messages. While this value is strong , there is still small margin for error. Misclassified messages could lead to missed spam detection or user expperience. From the further analysis of misclasified messages, it can be assumed that it was due to inclusion of many new words out of the vocabulary or not having the complete message displayed in the SMS.Naive Bayes theorem assumes words in the message are independent, which might not always true and could cause misclassification of words. Further evaluation could be done using metrics such as precision, recall and F1-score to asses how well the model balances between identifying spam and avoiding misclassification.

































    














 

 
