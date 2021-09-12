# Notebook Imports


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

# Constants


```python
TOKEN_SPAM_PROB_FILE = 'SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/03_Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = 'SpamData/03_Testing/test-features.txt'
TEST_TARGET_FILE = 'SpamData/03_Testing/test-target.txt'

VOCAB_SIZE = 2500
```

# Load the Data


```python
# Features
X_test = np.loadtxt(TEST_FEATURE_MATRIX, delimiter=' ')
# Target
y_test = np.loadtxt(TEST_TARGET_FILE, delimiter=' ')
# Token Probabilities
prob_token_spam = np.loadtxt(TOKEN_SPAM_PROB_FILE, delimiter=' ')
prob_token_ham = np.loadtxt(TOKEN_HAM_PROB_FILE, delimiter=' ')
prob_all_tokens = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter=' ')
```


```python
X_test[:5]
```

# Calculating the Joint Probability

### The Dot Product


#### The dimensions of the dot product between X_test and prob_token_spam



```python
X_test.shape
```


```python
prob_token_spam.shape
```


```python
print('shape of the dot product is ', X_test.dot(prob_token_spam).shape)
```

## Set the Prior

$$P(Spam \, | \, X) = \frac{P(X \, | \, Spam) \, P(Spam)} {P(X)}$$


```python
PROB_SPAM = 0.3116
```

####  Calculate the log probabilities of the tokens given that the email was spam. 
####  This was stored in prob_token_spam 


```python
np.log(prob_token_spam)
```

## Joint probability in log format


```python
joint_log_spam = X_test.dot(np.log(prob_token_spam) - np.log(prob_all_tokens)) + np.log(PROB_SPAM)
```

#### Calculate the log probability that the emails are non-spam given their tokens. Store the result in a variable called joint_log_ham

$$P(Ham \, | \, X) = \frac{P(X \, | \, Ham) \, (1-P(Spam))} {P(X)}$$


```python
joint_log_ham = X_test.dot(np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1-PROB_SPAM)
```


```python
joint_log_ham[:5]
```

# Making Predictions

### Checking for the higher joint probability

$$P(Spam \, | \, X) \, > \, P(Ham \, | \, X)$$
<center>**OR**</center>
<br>
$$P(Spam \, | \, X) \, < \, P(Ham \, | \, X)$$

#### Create the vector of predictions, our $\hat y$  
#### Remember that spam emails should have the value 1 (true) and non-spam emails should have the value 0 (false). 
#### Store your results in a variable called `prediction`


```python
prediction = joint_log_spam > joint_log_ham
```


```python
prediction[-5:]*1
```


```python
y_test[-5:]
```

### Simplify

$$P(X \, | \, Spam) \, P(Spam) ≠  \frac{P(X \, | \, Spam) \, P(Spam)}{P(X)}$$


```python
joint_log_spam = X_test.dot(np.log(prob_token_spam)) + np.log(PROB_SPAM)
joint_log_ham = X_test.dot(np.log(prob_token_ham)) + np.log(1-PROB_SPAM)
```

# Metrics and Evaluation

## Accuracy


```python

correct_docs = (y_test == prediction).sum()
print('Docs classified correctly', correct_docs)
numdocs_wrong = X_test.shape[0] - correct_docs
print('Docs classified incorrectly', numdocs_wrong)
```


```python
    # This is Number of the Accuracy
    # this is the fractin of the e-mil is classify correctly
    # collect prediction / total number of the prediction 
correct_docs/len(X_test)
```


```python
    # This is wrong prediction part
fraction_wrong = numdocs_wrong/len(X_test)
print('Fraction classified incorrectly is {:.2%}'.format(fraction_wrong))
print('Accuracy of the model is {:.2%}'.format(1-fraction_wrong))
```

## Visualising the Results

## The Decision Boundary


```python
# Chart Styling Info
yaxis_label = 'P(X | Spam)'
xaxis_label = 'P(X | Nonspam)'

linedata = np.linspace(start=-14000, stop=1, num=1000)
```


```python
# The boundar will divide the chart into 2 boundary
# The spam and non spam

plt.figure(figsize=(11, 7))
plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=25)
# Draw the line to seperate
plt.plot(linedata, linedata, color='orange')

plt.show()
```


```python
plt.figure(figsize=(16, 7))

# Chart Nr 1:
plt.subplot(1, 2, 1)

plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=25)
plt.plot(linedata, linedata, color='orange')

# Chart Nr 2:
plt.subplot(1, 2, 2)

plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=3)
plt.plot(linedata, linedata, color='orange')

plt.show()
```


```python
sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, size=6.5, fit_reg=False, legend=False,
          scatter_kws={'alpha': 0.5, 's': 25}, hue=labels, markers=['o', 'x'], palette='hls')

plt.xlim([-2000, 1])
plt.ylim([-2000, 1])

plt.plot(linedata, linedata, color='black')

plt.legend(('Decision Boundary', 'Nonspam', 'Spam'), loc='lower right', fontsize=14)

sns.plt.show()
```

### False Positives and False Negatives


```python
np.unique(prediction, return_counts=True)
```


```python
true_pos = (y_test == 1) & (prediction == 1)
```


```python
true_pos.sum()
```

#### Create a numpy array that measures the False Positives for each datapoint. 
#### Call this variable ```false_pos```. Then work out how many false positives there were. 
#### Do the same for the false negatives. 
#### Store those in a variable called ```false_neg```


```python
false_pos = (y_test == 0) & (prediction == 1)
false_pos.sum()
```


```python
false_neg = (y_test == 1) & (prediction == 0)
false_neg.sum()
```

## Recall Score

#### Calculate the recall score. Store it in a variable called ```recall_score```. 
#### Print the value of the recall score as a percentage rounded to two decimal places.


```python
recall_score = true_pos.sum() / (true_pos.sum() + false_neg.sum())
print('Recall score is {:.2%}'.format(recall_score))
```

## Precision Score

#### Calculate the precision of our naive bayes model. 
#### Store the result in a variable called ```precision_score```. 
#### Print out the precision as a decimal number rounded to three decimal places. 


```python
precision_score = true_pos.sum() / (true_pos.sum() + false_pos.sum())
print('Precision score is {:.3}'.format(precision_score))
```

## F-Score or F1 Score


```python
f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
print('F Score is {:.2}'.format(f1_score))
```
