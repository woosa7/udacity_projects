#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    The list of all the emails from Sara are in the from_sara list likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which you downloaded/unpacked in Part 0 of the first mini-project. 
    If you have not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:

        temp_counter += 1
        if temp_counter % 1000 == 0:  # total 17578
            print(temp_counter)

        # path = os.path.join('..', path[:-1])
        path = path[:-1]
        email = open(path, "r")

        ### use parseOutText to extract the text from the opened email
        parsed_text = parseOutText(email)

        ### use str.replace() to remove any instances of the words
        # drop_word = ["sara", "shackleton", "chris", "germani"]
        drop_word = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"] # remove outlier
        for word in drop_word:
            parsed_text = parsed_text.replace(word, "")

        word_data.append(parsed_text)

        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name.lower() == "sara":
            from_data.append(0)
        elif name.lower() == "chris":
            from_data.append(1)

        email.close()

print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )

# print(word_data[152])


### TfIdf vectorization
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
word_vec = vectorizer.fit_transform(word_data)

print(word_vec.shape)

print(len(vectorizer.get_feature_names()))

# print(vectorizer.get_feature_names()[34596])

