# Notebook Imports


```python
from os import walk
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

import numpy as np

from sklearn.model_selection import train_test_split

%matplotlib inline
```

# Constant 


```python
### this file for testing ###
EXAMPLE_FILE = 'SpamData/SpamData/01_Processing/practice_email.txt'
SPAM_1_PATH = 'SpamData/SpamData/01_Processing/spam_assassin_corpus/spam_1'
SPAM_2_PATH = 'SpamData/SpamData/01_Processing/spam_assassin_corpus/spam_2'
EASY_NONSPAM_1_PATH = 'SpamData/SpamData/01_Processing/spam_assassin_corpus/easy_ham_1'
EASY_NONSPAM_2_PATH = 'SpamData/SpamData/01_Processing/spam_assassin_corpus/easy_ham_2'

SPAM_CAT = 1
HAM_CAT = 0
VOCAB_SIZE = 2500

DATA_JSON_FILE = 'SpamData/SpamData/01_Processing/email-text-data.json'
WORD_ID_FILE = 'SpamData/SpamData/01_Processing/word-by-id.csv'

TRAINING_DATA_FILE = 'SpamData/SpamData/02_Training/train-data.txt'
TEST_DATA_FILE = 'SpamData/SpamData/02_Training/test-data.txt'
```

# Gather the Data


# Generator Functions

#### use generator function to loop all the file in dictionary that hold onto the spam e-mails
#### pass one e-mail at the time



```python
import sys
sys.getfilesystemencoding()
```


```python
###  This is for the purpose of testing ###
stream = open(EXAMPLE_FILE, encoding='latin-1')

is_body = False
lines = []

for line in stream:
    if is_body:
        lines.append(line)
    elif line == '\n':
        is_body = True

stream.close()

email_body = '\n'.join(lines)
print(email_body)
```

## E-mail body extraction 


```python
# purpose : this function extact the body from the e-mail
#           and extract file name
#           walk function where operating system come in
#           the walk func generate the file name 
#           by walking the tree from the top to the bottom
#           and yeld it
#           directly path - root
#           direcly name  - directly name
#           file name
def email_body_generator(path):
        for root, dirnames, filenames in walk(path):
            for file_name in filenames:
                filepath = join(root, file_name)
                stream = open(filepath, encoding='latin-1')
                is_body = False
                lines = []
                for line in stream:
                        if is_body:
                            lines.append(line)
                        elif line == '\n':
                            is_body = True

                stream.close()

                email_body = '\n'.join(lines)
            
                yield file_name, email_body
```


```python
# purospe : the generator function is called iniside the loop
#           to get data frame
def df_from_directory(path, classification):
        rows = []
        row_names = []
        for file_name, email_body in email_body_generator(path):
            rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
            row_names.append(file_name)
        return pd.DataFrame(rows, index=row_names)
```


```python
    # call df from directory function 
    # and create dataframe of the spam e-amil 
spam_emails = df_from_directory(SPAM_1_PATH,SPAM_CAT)

    # to extact all of the spam_2 path that contain spam e-mail
spam_emails = spam_emails.append(df_from_directory(SPAM_2_PATH, SPAM_CAT))

```


```python
    # call df from directory function 
    # and create dataframe of the non-spam e-amil 
ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)

    # to extact all of the spam_2 path that contain non-spam e-mail
ham_emails = ham_emails.append(df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT))


```


```python
    # create data frame that hold all the e-mail
data = pd.concat([spam_emails, ham_emails])
#print('Shape of entire dataframe is ', data.shape)
#data.head()

# As the resul from this section 
# I take 5800 files from the local disk
# and convert them into data pandas frame
# to work with 

print('Shape of entire dataframe is ', data.shape)
data.head()
```

# Data Cleaning  
## Checking for missing values


```python
    # check if any message bodies are null => empty
data['MESSAGE'].isnull().values.any()
# After checking 
# there is no missing value in the message column
```


```python
    # check if there are empty emails (string length zero)
(data.MESSAGE.str.len() == 0).any()
# There are some email that has string length of 0
```


```python
(data.MESSAGE.str.len() == 0).sum()
# There are 3 string of length 0
```


```python
    # check the number of entries with null/None values?
data.MESSAGE.isnull().sum()
# There are no e-mail that has null value
```

## Locate empty emails


```python
data[data.MESSAGE.str.len() == 0].index
# the empty e-mail is the e-mail file
# but they are system file
```

## Remove System File Entries from Dataframe


```python
# droping the e-mail that has the file name cmds
data.drop(['cmds'], inplace=True)
```

# Add Document IDs to Track Emails in Dataset


```python
    # the range of data frame
document_ids = range(0, len(data.index))

    # create new column for the index as the id
data['DOC_ID'] = document_ids
```


```python
    # crate the column for this file name
data['FILE_NAME'] = data.index
data.set_index('DOC_ID', inplace=True)

```

# Save to File using Pandas


```python
    # use jason file format
data.to_json(DATA_JSON_FILE)
```

# Explore and Visualise the data

# Number of Spam Messages Visualised (Pie Charts)


```python
    # the number of the spam and non - spam message
    # store in the varible

amount_of_spam = data.CATEGORY.value_counts()[1]
amount_of_ham = data.CATEGORY.value_counts()[0]
```


```python
    # Pie chart
category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%', colors=custom_colours, explode=[0, 0.1])
plt.show()
```

# Prepare the e-mail for Baye's Classify work

# Natural Language Processing

####    to prepare the text for learing algorithm
####    convert to the form that algoritimn can understand

# Text - Pre-Processing 

### Download the NLTK Resources (Tokenizer & Stopwords)




```python
nltk.download('punkt')
```


```python
nltk.download('stopwords')
```

## Tokenising 
####  (spliting the words in the sentence into individual words)

### Removing Stop Words
#### The stop word is not quite naive as classify as spam


```python
# if use list it gonna spend more time than set
    # set has not order
# use the set to visialse whether each of the word is stop word
stop_words = set(stopwords.words('english'))
```

###  Word Stems and Stemminng


```python
stemmer = SnowballStemmer('english')
```

### Removing Punctuation


```python

```


```python
### Removing HTML tags from Emails  ###
```


```python
                    # index name, column name
#data.at[2, 'MESSAGE']
soup = BeautifulSoup(data.at[2, 'MESSAGE'], 'html.parser')
print(soup.prettify())
```


```python
# remove the html text
soup.get_text()
```

### Functions for Email Processing


```python
# purpopse : to make all of the sentence in the body part of the e-mail to be lower case
#            and reomove the punctuation,and stop word
#            retun the anyswer in the array
def clean_message(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Converts to Lower Case and splits up the words(tokening)
    words = word_tokenize(message.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and and is not punctuation
        if word not in stop_words and word.isalpha():
            # append the answer in the filter
            filtered_words.append(stemmer.stem(word))
    
    return filtered_words
```


```python
### for testing
clean_message(email_body)
```


```python
def clean_msg_no_html(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Remove HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    
    # Converts to Lower Case and splits up the words
    words = word_tokenize(cleaned_text.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
        # filtered_words.append(word) 
    
    return filtered_words
```


```python
### for testing
clean_msg_no_html(data.at[2, 'MESSAGE'])
```

# Apply Cleaning and Tokenisation to all messages

### Slicing Dataframes and Series & Creating Subsets




```python
%%time

# Remove the Html tag to all of the message
# use apply() on all the messages in the dataframe
nested_list = data.MESSAGE.apply(clean_msg_no_html)
```

### Using Logic to Slice Dataframes


```python
# create two variables (doc_ids_spam, doc_ids_ham) which 
# hold onto the indices for the spam and the non-spam emails respectively. 
doc_ids_spam = data[data.CATEGORY == 1].index
doc_ids_ham = data[data.CATEGORY == 0].index
```


```python
doc_ids_ham
```

### Subsetting a Series with an Index


```python
nested_list_ham = nested_list.loc[doc_ids_ham]
```


```python
nested_list_spam = nested_list.loc[doc_ids_spam]
```


```python
# use python list comprehension and then find the total number of 
# words in our cleaned dataset of spam email bodies.
flat_list_ham = [item for sublist in nested_list_ham for item in sublist]
normal_words = pd.Series(flat_list_ham).value_counts()

normal_words.shape[0] # total number of unique words in the non-spam messages
```


```python
# use python list comprehension and then find the total number of 
# words in our cleaned dataset of spam email bodies.
flat_list_spam = [item for sublist in nested_list_spam for item in sublist]
spammy_words = pd.Series(flat_list_spam).value_counts()

spammy_words.shape[0] # total number of unique words in the spam messages
```

## Generate Vocabulary & Dictionary


```python
# 2500 most frequency word gonna form our vocaburary
stemmed_nested_list = data.MESSAGE.apply(clean_msg_no_html)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]
```


```python
# generate panda serie and use value count methd
unique_words = pd.Series(flat_stemmed_list).value_counts()
```


```python
    # Create subset of the series called 'frequent_words' that only contains
    # the most common 2,500 words out of the total. 
    # vocab size = 2500
frequent_words = unique_words[0:VOCAB_SIZE]

```

## Create Vocabulary DataFrame with a WORD_ID


```python
# assing the word id in each word
# store answer in the list and store the the word_ids
# store it in the data frame
word_ids = list(range(0, VOCAB_SIZE))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'WORD_ID'
# this is the vocabury that I will train our classifier
```

##  Save the Vocabulary as a CSV File


```python
vocab.to_csv(WORD_ID_FILE, index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)
```

## Generate Features & a Sparse Matrix

### Creating a DataFrame with one Word per Column

##### (The sparse matrix only include the row which have the word occurs in the e-mail)
##### (It will remove empty matrix)


```python
    # working with stem_nested list
    # each row contain the list of word for each document in the e-mail
    # the stem_nested list is the list
    # will convert in to list that contain the list by to list
    # and convert in to dataframe
word_columns_df = pd.DataFrame.from_records(stemmed_nested_list.tolist())

```

### Splitting the Data into a Training and Testing Dataset


```python
    # split the data into a training and testing set
    # Set the test size at 30%. 
    # The training data is comprise of 4057 emails. 
    # Use a seed value of 42 to shuffle the data. 

X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY,
                                                   test_size=0.3, random_state=42)
```


```python
# print('Nr of training samples', X_train.shape[0])
# print('Fraction of training set', X_train.shape[0] / word_columns_df.shape[0])

```


```python
X_train.index.name = X_test.index.name = 'DOC_ID'

```

### Create a Sparse Matrix for the Training Data


```python
word_index = pd.Index(vocab.VOCAB_WORD)
```


```python
    # take x-train to create the sparse matrix
def make_sparse_matrix(df, indexed_words, labels):
        """
        Returns sparse matrix as dataframe.
    
        df: A dataframe with words in the columns with a document id as an index (X_train or X_test)
        indexed_words: index of words ordered by word id
        labels: category as a series (y_train or y_test)
        """
    
        nr_rows = df.shape[0]
        nr_cols = df.shape[1]
        word_set = set(indexed_words)
        dict_list = []
        
        for i in range(nr_rows):
            for j in range(nr_cols):
            
                word = df.iat[i, j]
                if word in word_set:
                    doc_id = df.index[i]
                    word_id = indexed_words.get_loc(word)
                    # y value at document id
                    category = labels.at[doc_id]
                
                    item = {'LABEL': category, 'DOC_ID': doc_id,
                       'OCCURENCE': 1, 'WORD_ID': word_id}
                
                    dict_list.append(item)
                # data frame that create from the loop
        return pd.DataFrame(dict_list)
```


```python
%%time
sparse_train_df = make_sparse_matrix(X_train, word_index, y_train)
# pass a lot of data go through the data frame
```

### Combine Occurrences with the Pandas groupby() Method


```python
    # grouping the word by e-mail
train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
```


```python
train_grouped = train_grouped.reset_index()
```

### Save Training Data as .txt File


```python
np.savetxt(TRAINING_DATA_FILE, train_grouped, fmt='%d')
```

## Create a Sparse Matrix for the Testing Data¶


```python
    # create a sparse matrix for the test data.
%%time
sparse_test_df = make_sparse_matrix(X_test, word_index, y_test)
```


```python
    # group the occurrences of the same word in the same email.¶
test_grouped = sparse_test_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum().reset_index()
```


```python
    # save the data as a .txt file.¶
np.savetxt(TEST_DATA_FILE, test_grouped, fmt='%d')
```

## Checking


```python
    #started with 5796 emails. 
    # I split it into 4057 emails for training and 1739 emails for testing.
    # check for the number of the emails that were included in the testing .txt file
    #Count the number in the test_grouped DataFrame. 
# After splitting and shuffling our data, how many emails were included in the X_test DataFrame? 
# Is the number the same? 
# If not, which emails were excluded and why? Compare the DOC_ID values to find out.
```


```python
train_doc_ids = set(train_grouped.DOC_ID)
test_doc_ids = set(test_grouped.DOC_ID)
```


```python
    # Excluded emails after pre-processing
set(X_test.index.values) - test_doc_ids 
    # the word is not the part of the vocabulary or it is html tag
```


```python

```


```python

```
