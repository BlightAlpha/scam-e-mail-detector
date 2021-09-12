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

# Reading Files


```python
    ## this is for the purpose of testing the function ###
stream = open(EXAMPLE_FILE, encoding='latin-1')
message = stream.read()
stream.close()

print(type(message))
print(message)
```

    <class 'str'>
    From exmh-workers-admin@redhat.com  Thu Aug 22 12:36:23 2002
    Return-Path: <exmh-workers-admin@spamassassin.taint.org>
    Delivered-To: zzzz@localhost.netnoteinc.com
    Received: from localhost (localhost [127.0.0.1])
    	by phobos.labs.netnoteinc.com (Postfix) with ESMTP id D03E543C36
    	for <zzzz@localhost>; Thu, 22 Aug 2002 07:36:16 -0400 (EDT)
    Received: from phobos [127.0.0.1]
    	by localhost with IMAP (fetchmail-5.9.0)
    	for zzzz@localhost (single-drop); Thu, 22 Aug 2002 12:36:16 +0100 (IST)
    Received: from listman.spamassassin.taint.org (listman.spamassassin.taint.org [66.187.233.211]) by
        dogma.slashnull.org (8.11.6/8.11.6) with ESMTP id g7MBYrZ04811 for
        <zzzz-exmh@spamassassin.taint.org>; Thu, 22 Aug 2002 12:34:53 +0100
    Received: from listman.spamassassin.taint.org (localhost.localdomain [127.0.0.1]) by
        listman.redhat.com (Postfix) with ESMTP id 8386540858; Thu, 22 Aug 2002
        07:35:02 -0400 (EDT)
    Delivered-To: exmh-workers@listman.spamassassin.taint.org
    Received: from int-mx1.corp.spamassassin.taint.org (int-mx1.corp.spamassassin.taint.org
        [172.16.52.254]) by listman.redhat.com (Postfix) with ESMTP id 10CF8406D7
        for <exmh-workers@listman.redhat.com>; Thu, 22 Aug 2002 07:34:10 -0400
        (EDT)
    Received: (from mail@localhost) by int-mx1.corp.spamassassin.taint.org (8.11.6/8.11.6)
        id g7MBY7g11259 for exmh-workers@listman.redhat.com; Thu, 22 Aug 2002
        07:34:07 -0400
    Received: from mx1.spamassassin.taint.org (mx1.spamassassin.taint.org [172.16.48.31]) by
        int-mx1.corp.redhat.com (8.11.6/8.11.6) with SMTP id g7MBY7Y11255 for
        <exmh-workers@redhat.com>; Thu, 22 Aug 2002 07:34:07 -0400
    Received: from ratree.psu.ac.th ([202.28.97.6]) by mx1.spamassassin.taint.org
        (8.11.6/8.11.6) with SMTP id g7MBIhl25223 for <exmh-workers@redhat.com>;
        Thu, 22 Aug 2002 07:18:55 -0400
    Received: from delta.cs.mu.OZ.AU (delta.coe.psu.ac.th [172.30.0.98]) by
        ratree.psu.ac.th (8.11.6/8.11.6) with ESMTP id g7MBWel29762;
        Thu, 22 Aug 2002 18:32:40 +0700 (ICT)
    Received: from munnari.OZ.AU (localhost [127.0.0.1]) by delta.cs.mu.OZ.AU
        (8.11.6/8.11.6) with ESMTP id g7MBQPW13260; Thu, 22 Aug 2002 18:26:25
        +0700 (ICT)
    From: Robert Elz <kre@munnari.OZ.AU>
    To: Chris Garrigues <cwg-dated-1030377287.06fa6d@DeepEddy.Com>
    Cc: exmh-workers@spamassassin.taint.org
    Subject: Re: New Sequences Window
    In-Reply-To: <1029945287.4797.TMDA@deepeddy.vircio.com>
    References: <1029945287.4797.TMDA@deepeddy.vircio.com>
        <1029882468.3116.TMDA@deepeddy.vircio.com> <9627.1029933001@munnari.OZ.AU>
        <1029943066.26919.TMDA@deepeddy.vircio.com>
        <1029944441.398.TMDA@deepeddy.vircio.com>
    MIME-Version: 1.0
    Content-Type: text/plain; charset=us-ascii
    Message-Id: <13258.1030015585@munnari.OZ.AU>
    X-Loop: exmh-workers@spamassassin.taint.org
    Sender: exmh-workers-admin@spamassassin.taint.org
    Errors-To: exmh-workers-admin@spamassassin.taint.org
    X-Beenthere: exmh-workers@spamassassin.taint.org
    X-Mailman-Version: 2.0.1
    Precedence: bulk
    List-Help: <mailto:exmh-workers-request@spamassassin.taint.org?subject=help>
    List-Post: <mailto:exmh-workers@spamassassin.taint.org>
    List-Subscribe: <https://listman.spamassassin.taint.org/mailman/listinfo/exmh-workers>,
        <mailto:exmh-workers-request@redhat.com?subject=subscribe>
    List-Id: Discussion list for EXMH developers <exmh-workers.spamassassin.taint.org>
    List-Unsubscribe: <https://listman.spamassassin.taint.org/mailman/listinfo/exmh-workers>,
        <mailto:exmh-workers-request@redhat.com?subject=unsubscribe>
    List-Archive: <https://listman.spamassassin.taint.org/mailman/private/exmh-workers/>
    Date: Thu, 22 Aug 2002 18:26:25 +0700
    
    
    Dear Mr Still
    
    Good tidings to you and all your staff for the festive season ahead (Christmas).
    Now to the crux of the matter-in-hand: I am a fully qualified Santa Claus and am wondering whether you might consider me to run my own "Santa's Grotto" in your store.
    But WAIT! You're probably thinking: "What makes him so special?"
    Well, first of all, I have made several changes to the characterisation of Father Christmas. Rather than greeting the children with shouts of "Ho, ho, ho!" I prefer to whisper the phrase "Dependence is not unfathomable in this cruel world we live in". In addition, my gifts are ALL hand-made, ranging from felt hoops to vanilla-pod holders.
    You will note also, from the enclosed sketch, that I have radically redesigned Santa's outfit and have renamed my character "Lord Buckles". Would you be interested in employing me? I promise NEVER to let you down.
    I look forward to hearing from you.
    
    Best wishes
    Robin Cooper
    [Excerpt from the book: The Timewaster Letters by Robin Cooper]
    


# Gather the Data


# Generator Functions

#### use generator function to loop all the file in dictionary that hold onto the spam e-mails
#### pass one e-mail at the time



```python
import sys
sys.getfilesystemencoding()
```




    'utf-8'




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

    
    
    Dear Mr Still
    
    
    
    Good tidings to you and all your staff for the festive season ahead (Christmas).
    
    Now to the crux of the matter-in-hand: I am a fully qualified Santa Claus and am wondering whether you might consider me to run my own "Santa's Grotto" in your store.
    
    But WAIT! You're probably thinking: "What makes him so special?"
    
    Well, first of all, I have made several changes to the characterisation of Father Christmas. Rather than greeting the children with shouts of "Ho, ho, ho!" I prefer to whisper the phrase "Dependence is not unfathomable in this cruel world we live in". In addition, my gifts are ALL hand-made, ranging from felt hoops to vanilla-pod holders.
    
    You will note also, from the enclosed sketch, that I have radically redesigned Santa's outfit and have renamed my character "Lord Buckles". Would you be interested in employing me? I promise NEVER to let you down.
    
    I look forward to hearing from you.
    
    
    
    Best wishes
    
    Robin Cooper
    
    [Excerpt from the book: The Timewaster Letters by Robin Cooper]
    


## E-mail body extraction 


```python
    # purpose : this function extact the body from the e-mail
    #           and extract file name
    # walk function where operating system come in
        # the walk func generate the file name 
        # by walking the tree from the top to the bottom
        # and yeld it
            # directly path - root
            # direcly name  - directly name
            # file na
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

    Shape of entire dataframe is  (5799, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MESSAGE</th>
      <th>CATEGORY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00001.7848dde101aa985090474a91ec93fcf0</th>
      <td>&lt;!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Tr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>00002.d94f1b97e48ed3b553b3508d116e6a09</th>
      <td>1) Fight The Risk of Cancer!\n\nhttp://www.adc...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>00003.2ee33bc6eacdb11f38d052c44819ba6c</th>
      <td>1) Fight The Risk of Cancer!\n\nhttp://www.adc...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>00004.eac8de8d759b7e74154f142194282724</th>
      <td>##############################################...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>00005.57696a39d7d84318ce497886896bf90d</th>
      <td>I thought you might like these:\n\n1) Slim Dow...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Data Cleaning  
## Checking for missing values


```python
    # check if any message bodies are null => empty
data['MESSAGE'].isnull().values.any()
# After checking 
# there is no missing value in the message column
```




    False




```python
    # check if there are empty emails (string length zero)
(data.MESSAGE.str.len() == 0).any()
# There are some email that has string length of 0
```




    True




```python
(data.MESSAGE.str.len() == 0).sum()
# There are 3 string of length 0
```




    3




```python
    # check the number of entries with null/None values?
data.MESSAGE.isnull().sum()
# There are no e-mail that has null value
```




    0



## Locate empty emails


```python
data[data.MESSAGE.str.len() == 0].index
# the empty e-mail is the e-mail file
# but they are system file
```




    Index(['cmds', 'cmds', 'cmds'], dtype='object')



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


    
![png](output_31_0.png)
    


# Prepare the e-mail for Baye's Classify work

# Natural Language Processing

####    to prepare the text for learing algorithm
####    convert to the form that algoritimn can understand

# Text - Pre-Processing 

### Download the NLTK Resources (Tokenizer & Stopwords)




```python
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/surapaphrompha/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True




```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/surapaphrompha/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True



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

    1) Fight The Risk of Cancer!
    
    http://www.adclick.ws/p.cfm?o=315&amp;s=pk007
    
    
    
    2) Slim Down - Guaranteed to lose 10-12 lbs in 30 days
    
    http://www.adclick.ws/p.cfm?o=249&amp;s=pk007
    
    
    
    3) Get the Child Support You Deserve - Free Legal Advice
    
    http://www.adclick.ws/p.cfm?o=245&amp;s=pk002
    
    
    
    4) Join the Web's Fastest Growing Singles Community
    
    http://www.adclick.ws/p.cfm?o=259&amp;s=pk007
    
    
    
    5) Start Your Private Photo Album Online!
    
    http://www.adclick.ws/p.cfm?o=283&amp;s=pk007
    
    
    
    Have a Wonderful Day,
    
    Offer Manager
    
    PrizeMama
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    If you wish to leave this list please use the link below.
    
    http://www.qves.com/trim/?zzzz@spamassassin.taint.org%7C17%7C308417
    



```python
# remove the html text
soup.get_text()
```




    "1) Fight The Risk of Cancer!\n\nhttp://www.adclick.ws/p.cfm?o=315&s=pk007\n\n\n\n2) Slim Down - Guaranteed to lose 10-12 lbs in 30 days\n\nhttp://www.adclick.ws/p.cfm?o=249&s=pk007\n\n\n\n3) Get the Child Support You Deserve - Free Legal Advice\n\nhttp://www.adclick.ws/p.cfm?o=245&s=pk002\n\n\n\n4) Join the Web's Fastest Growing Singles Community\n\nhttp://www.adclick.ws/p.cfm?o=259&s=pk007\n\n\n\n5) Start Your Private Photo Album Online!\n\nhttp://www.adclick.ws/p.cfm?o=283&s=pk007\n\n\n\nHave a Wonderful Day,\n\nOffer Manager\n\nPrizeMama\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nIf you wish to leave this list please use the link below.\n\nhttp://www.qves.com/trim/?zzzz@spamassassin.taint.org%7C17%7C308417\n\n\n"



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




    ['dear',
     'mr',
     'still',
     'good',
     'tide',
     'staff',
     'festiv',
     'season',
     'ahead',
     'christma',
     'crux',
     'fulli',
     'qualifi',
     'santa',
     'clau',
     'wonder',
     'whether',
     'might',
     'consid',
     'run',
     'santa',
     'grotto',
     'store',
     'wait',
     'probabl',
     'think',
     'make',
     'special',
     'well',
     'first',
     'made',
     'sever',
     'chang',
     'characteris',
     'father',
     'christma',
     'rather',
     'greet',
     'children',
     'shout',
     'ho',
     'ho',
     'ho',
     'prefer',
     'whisper',
     'phrase',
     'depend',
     'unfathom',
     'cruel',
     'world',
     'live',
     'addit',
     'gift',
     'rang',
     'felt',
     'hoop',
     'holder',
     'note',
     'also',
     'enclos',
     'sketch',
     'radic',
     'redesign',
     'santa',
     'outfit',
     'renam',
     'charact',
     'lord',
     'buckl',
     'would',
     'interest',
     'employ',
     'promis',
     'never',
     'let',
     'look',
     'forward',
     'hear',
     'best',
     'wish',
     'robin',
     'cooper',
     'excerpt',
     'book',
     'timewast',
     'letter',
     'robin',
     'cooper']




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




    ['fight',
     'risk',
     'cancer',
     'http',
     'slim',
     'guarante',
     'lose',
     'lb',
     'day',
     'http',
     'get',
     'child',
     'support',
     'deserv',
     'free',
     'legal',
     'advic',
     'http',
     'join',
     'web',
     'fastest',
     'grow',
     'singl',
     'commun',
     'http',
     'start',
     'privat',
     'photo',
     'album',
     'onlin',
     'http',
     'wonder',
     'day',
     'offer',
     'manag',
     'prizemama',
     'wish',
     'leav',
     'list',
     'pleas',
     'use',
     'link',
     'http',
     'zzzz']



# Apply Cleaning and Tokenisation to all messages

### Slicing Dataframes and Series & Creating Subsets




```python
%%time

# Remove the Html tag to all of the message
# use apply() on all the messages in the dataframe
nested_list = data.MESSAGE.apply(clean_msg_no_html)
```

    /opt/anaconda3/lib/python3.8/site-packages/bs4/__init__.py:417: MarkupResemblesLocatorWarning: "http://www.post-gazette.com/columnists/20020905brian5
    " looks like a URL. Beautiful Soup is not an HTTP client. You should probably use an HTTP client like requests to get the document behind the URL, and feed that document to Beautiful Soup.
      warnings.warn(


    CPU times: user 46.8 s, sys: 449 ms, total: 47.2 s
    Wall time: 47.5 s


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




    Int64Index([1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905,
                ...
                5786, 5787, 5788, 5789, 5790, 5791, 5792, 5793, 5794, 5795],
               dtype='int64', name='DOC_ID', length=3900)



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




    20815




```python
# use python list comprehension and then find the total number of 
# words in our cleaned dataset of spam email bodies.
flat_list_spam = [item for sublist in nested_list_spam for item in sublist]
spammy_words = pd.Series(flat_list_spam).value_counts()

spammy_words.shape[0] # total number of unique words in the spam messages
```




    13242



## Generate Vocabulary & Dictionary


```python
# 2500 most frequency word gonna form our vocaburary
stemmed_nested_list = data.MESSAGE.apply(clean_msg_no_html)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]
```

    /opt/anaconda3/lib/python3.8/site-packages/bs4/__init__.py:417: MarkupResemblesLocatorWarning: "http://www.post-gazette.com/columnists/20020905brian5
    " looks like a URL. Beautiful Soup is not an HTTP client. You should probably use an HTTP client like requests to get the document behind the URL, and feed that document to Beautiful Soup.
      warnings.warn(



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
