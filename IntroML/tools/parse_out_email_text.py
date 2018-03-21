#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string


def stemString(inputString):
    stemmer = SnowballStemmer("english")

    tokens = inputString.split()
    singles = [stemmer.stem(token) for token in tokens]
    stemmed_string =  ' '.join(singles)

    return stemmed_string


def parseOutText(f):
    """ given an opened email file f, parse out all text below the metadata block at the top
        and return a string that contains all the words in the email (space-separated)
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")[1]
    words = ""

    if len(content) > 1:
        ### remove punctuation
        text_string = content.translate(str.maketrans("", "", string.punctuation))   # python 3.x
        # print(text_string)

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single space between each stemmed word)
        # words = text_string

        words = stemString(text_string)

    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()
