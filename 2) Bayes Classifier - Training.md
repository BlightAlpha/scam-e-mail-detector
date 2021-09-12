# Notebook Imports


```python
import pandas as pd
import numpy as np
```

# Constants


```python
TRAINING_DATA_FILE = 'SpamData/SpamData/02_Training/train-data.txt'
TEST_DATA_FILE = 'SpamData/SpamData/02_Training/test-data.txt'

TOKEN_SPAM_PROB_FILE = 'SpamData/SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/SpamData/03_Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = 'SpamData/SpamData/03_Testing/test-features.txt'
TEST_TARGET_FILE = 'SpamData/SpamData/03_Testing/test-target.txt'

VOCAB_SIZE = 2500
```

# Read and Load Features from .txt Files into NumPy Array


```python
                            # the relative path to the data file   data type
sparse_train_data = np.loadtxt(TRAINING_DATA_FILE, delimiter=' ', dtype=int)
```


```python
sparse_test_data = np.loadtxt(TEST_DATA_FILE, delimiter=' ', dtype=int)
```


```python
    # this is for checking 
print('Nr of rows in training file', sparse_train_data.shape[0])
print('Nr of rows in test file', sparse_test_data.shape[0])
print('Nr of emails in training file', np.unique(sparse_train_data[:, 0]).size)
print('Nr of emails in test file', np.unique(sparse_test_data[:, 0]).size)

```

### Create an Empty DataFrame


```python
column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
```


```python
index_names = np.unique(sparse_train_data[:, 0])
```


```python
full_train_data = pd.DataFrame(index=index_names, columns=column_names)
full_train_data.fillna(value=0, inplace=True)
```

# Create a Full Matrix from a Sparse Matrix


```python
def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, cat_idx=2, freq_idx=3):
    """
    Form a full matrix from a sparse matrix. Return a pandas dataframe. 
    Keyword arguments:
    sparse_matrix -- numpy array
    nr_words -- size of the vocabulary. Total number of tokens. 
    doc_idx -- position of the document id in the sparse matrix. Default: 1st column
    word_idx -- position of the word id in the sparse matrix. Default: 2nd column
    cat_idx -- position of the label (spam is 1, nonspam is 0). Default: 3rd column
    freq_idx -- position of occurrence of word in sparse matrix. Default: 4th column
    """
    column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
    doc_id_names = np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(index=doc_id_names, columns=column_names)
    full_matrix.fillna(value=0, inplace=True)
            # the number of row in sparse matrix
    for i in range(sparse_matrix.shape[0]):
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurrence = sparse_matrix[i][freq_idx]
        
        full_matrix.at[doc_nr, 'DOC_ID'] = doc_nr
        full_matrix.at[doc_nr, 'CATEGORY'] = label
        full_matrix.at[doc_nr, word_id] = occurrence
    
    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix
    
```


```python
%%time
full_train_data = make_full_matrix(sparse_train_data, VOCAB_SIZE)
```


## Training the Naive Bayes Model

### Calculating the Probability of Spam


```python
# Calculate the probability of spam - the percent of spam messages in the training
# dataset. Store this value in a variable called prob_spam
```


```python
prob_spam = full_train_data.CATEGORY.sum() / full_train_data.CATEGORY.size
print('Probability of spam is', prob_spam)
```


## Total Number of Words / Tokens


```python
    # take all of the column except for the category
full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
```


```python
email_lengths = full_train_features.sum(axis=1)
```


```python
total_wc = email_lengths.sum()
```


## Number of Tokens in Spam & Ham Emails


```python
# Create a subset of the email_lengths series that only contains the spam messages 
#Call the subset spam_lengths. Then count the total number of words that occur in spam emails.

```


```python
spam_lengths = email_lengths[full_train_data.CATEGORY == 1]
spam_wc = spam_lengths.sum()
```


```python
# Create a subset called ham_lengths. 
# Then count the total number of words that occur in the ham emails.
```


```python
ham_lengths = email_lengths[full_train_data.CATEGORY == 0]
nonspam_wc = ham_lengths.sum()
```


## Summing the Tokens Occuring in Spam


```python
train_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 1]
```


```python
summed_spam_tokens = train_spam_tokens.sum(axis=0) + 1
```


## Summing the Tokens Occuring in Ham


```python
# Sum the tokens that occur in the 
# non-spam messages. Store the values in a variable called summed_ham_tokens

train_ham_tokens = full_train_features.loc[full_train_data.CATEGORY == 0]
summed_ham_tokens = train_ham_tokens.sum(axis=0) + 1
```


```python
train_ham_tokens[2499].sum() + 1
```


## P(Token | Spam) - Probability that a Token Occurs given the Email is Spam



```python
prob_tokens_spam = summed_spam_tokens / (spam_wc + VOCAB_SIZE)
```


## P(Token | Ham) - Probability that a Token Occurs given the Email is Nonspam



```python
prob_tokens_nonspam = summed_ham_tokens / (nonspam_wc + VOCAB_SIZE)
```


##  P(Token) - Probability that Token Occurs



```python
prob_tokens_all = full_train_features.sum(axis=0) / total_wc
```


## Save the Trained Model


```python
np.savetxt(TOKEN_SPAM_PROB_FILE, prob_tokens_spam)
np.savetxt(TOKEN_HAM_PROB_FILE, prob_tokens_nonspam)
np.savetxt(TOKEN_ALL_PROB_FILE, prob_tokens_all)
```


## Prepare Test Data



```python
# Create a full matrix from the sparse_test_data. 
# Time the function call. How long does it take 
# Separate the features and the target values. 
# Save these as separate .txt files: a TEST_TARGET_FILE and a TEST_FEATURE_MATRIX file.
```


```python
%%time
full_test_data = make_full_matrix(sparse_test_data, nr_words=VOCAB_SIZE)
```


```python
X_test = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
y_test = full_test_data.CATEGORY
```


```python
np.savetxt(TEST_TARGET_FILE, y_test)
np.savetxt(TEST_FEATURE_MATRIX, X_test)
```


```python

```
