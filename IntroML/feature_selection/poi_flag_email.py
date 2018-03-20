#!/usr/bin/python

import sys
import reader
import poi_emails


def getToFromStrings(f):
    '''
    The imported reader.py file contains functions that we've created to help parse e-mails from the corpus.
    .getAddresses() reads in the opening lines of an e-mail to find the To: From: and CC: strings,
    .parseAddresses() line takes each string and extracts the e-mail addresses as a list.
    '''
    f.seek(0)
    to_string, from_string, cc_string = reader.getAddresses(f)
    to_emails = reader.parseAddresses(to_string)
    from_emails = reader.parseAddresses(from_string)
    cc_emails = reader.parseAddresses(cc_string)

    return to_emails, from_emails, cc_emails


### POI flag an email

def poiFlagEmail(f):
    """ given an email file f,
        return a trio of booleans for whether that email is to, from, or cc'ing a poi """

    to_emails, from_emails, cc_emails = getToFromStrings(f)

    ### poi_emails.poiEmails() returns a list of all POIs' email addresses.
    poi_email_list = poi_emails.poiEmails()

    to_poi = False
    from_poi = False
    cc_poi = False

    if to_emails:
        ctr = 0
        while not to_poi and ctr < len(to_emails):
            if to_emails[ctr] in poi_email_list:
                to_poi = True
            ctr += 1

    if cc_emails:
        ctr = 0
        while not cc_poi and ctr < len(cc_emails):
            if cc_emails[ctr] in poi_email_list:
                cc_poi = True
            ctr += 1

    if from_emails:
        ctr = 0
        while not from_poi and ctr < len(from_emails):
            if from_emails[ctr] in poi_email_list:
                from_poi = True
            ctr += 1


    return to_poi, from_poi, cc_poi


# ===============================================================
import os
import sys
import zipfile
from poi_flag_email import poiFlagEmail, getToFromStrings

data_dict = {}

with zipfile.ZipFile('emails.zip', "r") as z:
    z.extractall()

for email_message in os.listdir("emails"):
    if email_message == ".DS_Store":
        continue
    message = open(os.getcwd() + "/emails/" + email_message, "r")
    to_addresses, from_addresses, cc_addresses = getToFromStrings(message)

    to_poi, from_poi, cc_poi = poiFlagEmail(message)

    for recipient in to_addresses:
        # initialize counter
        if recipient not in data_dict:
            data_dict[recipient] = {"from_poi_to_this_person": 0}
        # add to count
        if from_poi:
            data_dict[recipient]["from_poi_to_this_person"] += 1

    message.close()


for item in data_dict:
    print(item, data_dict[item])

