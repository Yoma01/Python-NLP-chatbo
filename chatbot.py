"""
To run this project
install all libraries needed
run on 64 bit python interpreter
makes sure to run py file in the same directory with all other dependent files like
the knowledge base, positive and negative training text file, data json file and pickle file
"""

#imports
import os
import pickle
import random
import re
import string
import en_core_web_sm
import wikipedia
import spacy
import nltk
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import NaiveBayesClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tag import pos_tag
from sklearn.metrics.pairwise import cosine_similarity

bot_name = "Miki"
mood = "good"
"""
likes = ""
dislikes = ""
email =""
genre =""
song =""
artist =""
"""
possible_greetings =("What's up.", "Hello.", "Bleep-Blop.", "What's good my friend.", "Hey.",
                 "Greetings.", "HIIIII!!!!!!!",)
print(random.choice(possible_greetings))

introduction = "BOT: It's your favorite music-bot, MIki!!!. I am a chat-bot that answers you music questions and tell you about the various genres of music. " \
                   "I'm so excited to learn more about you and your music preference!!"

print(introduction)
print("BOT: What is your name")
user_name = input("user: ")
bot_template = "BOT : {0}"
user_template = user_name + " : {0}"

"""
This function introduces the bot and tells the user about the chatbot
It gives the user information on how to operate the bot
It also collects the users information and stores it in a json file
"""
def greetings():
    """
    #opens json file for storing data
    with open("data.json", "r") as f:
        data = json.load(f)
    # collects new users information
    if user_name not in data:
        print(bot_template.format("What is your email"))
        email = input( user_name+ ": ")
        print(bot_template.format("What kind of music genre do you listen to?"))
        genre = input(user_name+ ": ")
        print(bot_template.format("What is your favorite song?"))
        song = input(user_name+ ": ")
        print(bot_template.format("What is your favorite artist?"))
        artist = input(user_name+ ": ")
        data[user_name] = [email, genre, song, artist, likes, dislikes]
        with open("data.json", "w+") as f:
            json.dump(data, f)
    # shows old users information
    else:
        print("BOT: Welcome back", user_name)
        print("BOT: Your email is: ", data[user_name][0])
        print("BOT: Your kind of music genre is: ", data[user_name][1])
        print("BOT: Your favorite song is: ", data[user_name][2])
        print("BOT: Your favorite artist is: ", data[user_name][3])
        """
    # random responses given to engage the user
    responses = {
        "yes": ["cool cool cool :)", "AHHH YEAHH!!", "yayyyy!!!, now we are talking", "you've come to the right place",
                "AMAZING!!!, lets begin"],
        "no": ["You can't be serious.", "I don't respect you, just kidding",
               "Ummmm, this conversation is over, just kidding",
               "Very interesting response"]}
    starter = ("Amazing !!", "Good job!!", "Now that's done with,", "Yayy!!")
    print("BOT: " + random.choice(starter))
    print("BOT: So do you enjoy listening to music?")
    answer = yes_no(user_name + ": ")
    print("BOT: " + random.choice(responses[answer.lower()]))
    print("BOT: Type exit, goodbye or quit to end the chat")
    print("BOT: Alright, so what's up?")
    return user_name

"""
This function helps in receiving either yes or no answers from the greeting function and other function
It loops until a yes or no answer is given
"""
def yes_no(prompt):
    # Get input
    answer = input(prompt).strip().lower()
    # While not yes or no, repeat
    while 'yes' != answer and 'no' != answer:
        print("I don't really understand you. Please answer it in a yes or no format.")
        answer = input(prompt).lower()
    # Return results
    return "yes" if 'yes' in answer else "no"

"""
This function helps to remove unnecessary data and process the tokens so that
they can be passed to other functions
"""
def process(tokens):
    processed = []
    for t, tag in pos_tag(tokens):
        t = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', t)
        t = re.sub("(@[A-Za-z0-9_]+)", "", t)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
    # use wordNetlemmatizer to lemmatize the word
    lemma = WordNetLemmatizer()
    t = lemma.lemmatize(t,pos)
    # if the token is valid
    if len(t) > 0 and t not in string.punctuation and t.lower() not in stopwords.words('english'):
        processed.append(t.lower())
    # return processed tokens
    return processed

"""
This functions helps to perform sentement analysis making use of positive and negative tweets
as samples to build and train the model
"""
def train():
    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    pos_tokens = twitter_samples.tokenized('positive_tweets.json')
    neg_tokens = twitter_samples.tokenized('negative_tweets.json')
    # list for tokens
    pos_list = []
    neg_list = []
    # process data
    for token in pos_tokens:
        pos_list.append(process(token))
    for token in neg_tokens:
        neg_list.append(process(token))
    # get tweets with helper function
    pos_model_tweets = ret_tweet(pos_list)
    neg_model_tweets = ret_tweet(neg_list)
    # label tweets
    pos_set = [(tt_dict, "Positive") for tt_dict in pos_model_tweets]
    neg_set = [(tt_dict, "Negative") for tt_dict in neg_model_tweets]
    # combine sets
    pos_neg_set = pos_set + neg_set
    classifier = NaiveBayesClassifier.train(pos_neg_set)
    return classifier
# helper function to return tweets from tokens
def ret_tweet(t_list):
    for tokens in t_list:
        yield dict([token, True] for token in tokens)
# helper function to train the model and perform sentiment analysis
def ret_token(t_list):
    for tokens in t_list:
        for token in tokens:
            yield token


"""
This is a helper function to append values to a key in the chatbot dictionary
"""
def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        #add while loopto ask for user name
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value
"""
This function gets input from the user and determines if it is a
question with the aid of the ML function
"""
def getInput():
    # Get user input
    user_input = input(user_name +": ").strip().lower()
    # use spacy to understand input
    nlp = en_core_web_sm.load()
    doc = nlp(user_input)
    count = 0
    is_question = False
    w_count = -1000

    # For each word in the user_input
    for token in doc:
        # Is it a w tag such as WHAT,WHO,HOW,etc.
        if 'W' in token.tag_:
            w_count = count
        # Does a v tag follow a w tag, if so then it is a question
        if 'V' in token.tag_ and w_count == count - 1:
            is_question = True
        count += 1

    # If the statement didn't follow the structure
    if not is_question:
        is_question = ML_question(user_input)

    # Return boolean saying if it is a question and return user input
    return is_question, user_input

"""
This function makes use of training files to go ahead and examine questions that dont follow the rules
It makes use of machine learning by studying positive and negative training files to understand questions and 
determine if the input is a question
"""
def ML_question(input):
    # create a data frame for the user input
    d_f = pd.DataFrame([input], columns=['test'])

    # Create DF with positive training data
    file = open("train_question_positive.txt", "r", encoding='UTF-8')
    positive_train = file.readlines()
    positive_train = [sent.strip() for sent in positive_train]
    positive_df = pd.DataFrame(positive_train, columns=['sent'])
    positive_df['label'] = [1 for sent in positive_train]

    # Create DF with negative training data
    file = open("train_question_negative.txt", "r", encoding='UTF-8')
    negative_train = file.readlines()
    negative_train = [sent.strip() for sent in negative_train]
    negative_df = pd.DataFrame(negative_train, columns=['sent'])
    negative_df['label'] = [0 for sent in negative_train]

    # Combine the data set
    frames = [positive_df, negative_df]
    combined_df = pd.concat(frames, ignore_index=True, sort=False)

    # Get the features and the labels
    sentences = combined_df['sent'].values
    y = combined_df['label'].values

    # Vectorize the training data and the test data
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences)
    X_train = vectorizer.transform(sentences)
    X_test = vectorizer.transform(d_f['test'].values)

    # Train the model on the train data
    c_m = LogisticRegression()
    c_m.fit(X_train, y)

    # Predict the user_input
    probability = c_m.predict_proba(X_test[0])[0][1]

    # If high probability then is_question = true
    if probability >= 0.7:
        return True

    # If probability is near half then ask just to be sure.
    elif probability < 0.7 and probability >= 0.60:
        req = yes_no("Confirm if this is a question with yes or no")
        return True if req == "yes" else False

    # Otherwise it is not a question
    else:
        return False

"""
This function acesses the knowledge base and makes use of the information to aid with answering the queestions
"""
def genre_music(user_input):
    t_f = False
    count_list = []
    # get data from knowledge base about music facts
    #raw_data = wikipedia.page('Music genre')
    file = open('knowledgebase (1).txt', 'r', errors='ignore')
    raw_data = file.read().lower()
    #raw_data = raw_data.content
    # process raw data
    tokens = nltk.word_tokenize(raw_data)
    sentence_tokens = nltk.sent_tokenize(raw_data)
    sentence_tokens = [sent.replace("\n", " ").replace("\t", " ").replace("==", "").lower() for sent in sentence_tokens]
    # lemmatize the words
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in stopwords.words('english') and word.isalnum()]
    t_user = nltk.word_tokenize(user_input)
    t_user = [lemmatizer.lemmatize(word.lower()) for word in t_user if word not in stopwords.words('english')]
    for word in t_user:
        if tokens.count(word) > 0:
            count_list.append((word, tokens.count(word)))
            t_f = True
    if not t_f:
        # Return empty list and knowledge base
        return [], sentence_tokens
    else:
        # Sort the list to get the most relevant in the beginning and return it along with the knowledge base
        return sorted(count_list, key=lambda x: -x[1]), sentence_tokens


"""
This function tries to answer questions making use of NLP techniques to answer the user's questions
It gets the relevant terms in the knowledge base and checks if they are related to music and if not, it may ask for a music related fact
Cosine similarity will be performed on the knowledge base and the best answer will be determined and returned
"""
def response(user_input, user_name):
    checked, music_key = genre_music(user_input)
    # check if general questions or for music
    while len(checked) <= 0:
        print("BOT: enter a music related question in a simple words")
        user_input = input(user_name+": ").strip().lower()
        if "no" in user_input:
            return "BOT: cool cool cool. Alrighty, Let's continue."
        checked, music_key = genre_music(user_input)
    sent_tokens = music_key + nltk.sent_tokenize(user_input)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    cos_flat = cosine_similarity(tfidf[-1], tfidf)

    # Flatten the array and get the index with the closest similarity
    cos_flat = cos_flat.flatten()
    index = np.where(cos_flat == np.max(cos_flat[:-1]))[0][0]

    sent_t_f = False
    for word in checked:
        if word[0] in sent_tokens[index]:
            sent_t_f= True

    if not sent_t_f:
        responses = ["I don't know how to answer that but you can google it.", "coming up blank on that one, sorry about that!! ",
                                                                "Hmmm, I'm not too sure of that",
                      "good luck with that because I don't know the answer.", "Nope, nothing up here about that question"]
        return random.choice(responses)
    return sent_tokens[index]

"""
This function examines the users statement and performs sentiment analysis on it to give the best reply
to the question with the same sentiments
"""
def reply(user_name, user_input,model):
    # determines the sentiment
    sent_anal = process(word_tokenize(user_input))
    analysis = model.classify(dict([token, True] for token in sent_anal))
    # tries to determine negative sentiment
    neg_analysis = ["don't", "do not", "not", "dont"]
    for neg in neg_analysis:
        if neg in user_input:
            analysis = 'Negative'
    # tries to determine a command
    commands = ["tell me", "let me know", "talk to me", "want to know", "show me"]
    for comm in commands:
        # Then query the command and return the response
        if comm in user_input:
            answ = response(user_input, user_name)
            return answ

    if analysis == 'Positive':
        # Reply with a positive response
        positive_responses = ["Wow!! That's pretty cool.", "Awesome! Tell me more.", "I guess so.",
                              "Cool. But are going to ask a question?", "I'll keep track of that.",
                              "That's amazing","That's interesting","Well, that's good to know."]
        return random.choice(positive_responses)


    else:
        # Reply with a negative response
        negative_responses = ["I am sorry you feel that way.", "I appreciate your honesty.", "I empathize with you"]
        return random.choice(negative_responses)

def main():
    # train the model
    if not os.path.exists("train.pkl"):
        print("Chat bot is setting up")
        print("Please hold on....")
        # train with tweets for sentiment analysis
        SA_model = train()
        pickle.dump(SA_model,open("train.pkl","wb"))
    else:
        SA_model = pickle.load(open("train.pkl","rb"))
    # call greetings and get users information
    with open("data.json", "r") as f:
        data = json.load(f)
    # collects new users information
    if user_name not in data:
        print(bot_template.format("What is your email"))
        email = input( user_name+ ": ")
        print(bot_template.format("What kind of music genre do you listen to?"))
        genre = input(user_name+ ": ")
        print(bot_template.format("What is your favorite song?"))
        song = input(user_name+ ": ")
        print(bot_template.format("What is your favorite artist?"))
        artist = input(user_name+ ": ")

    # shows old users information
    else:
        print("BOT: Welcome back", user_name)
        print("BOT: Your email is: ", data[user_name][0])
        print("BOT: Your kind of music genre is: ", data[user_name][1])
        print("BOT: Your favorite song is: ", data[user_name][2])
        print("BOT: Your favorite artist is: ", data[user_name][3])
    greetings()
    is_question, user_input = getInput()
    # check if any exit word was entered
    while 'bye' not in user_input and'goodbye' not in user_input and 'quit' not in user_input and 'exit' not in user_input:
        if is_question:
            given_answer = response(user_input,user_name)
            print("BOT: "+ given_answer.capitalize())

        else:
            # Generate result and print it
            given_answer = reply(user_name, user_input, SA_model)
            print("BOT: ", given_answer)

        is_question, user_input = getInput()
    exit_responses = ["PEACE!", "Later!", "See ya!", "Bye.", "Take care", "Goodbye", "Hope you come back!",
                      "Sad to see you go"]
    # Print a goodbye message
    print("BOT: Before you leave, I will appreciate your feedback")
    print("BOT: What did you like about the chatbot")
    likes = input(user_name + ": ")
    print("BOT: What did you dislike about the chatbot")
    dislikes = input(user_name + ": ")

    if user_name not in data:
        data[user_name] = [email, genre, song, artist, likes, dislikes]
        with open("data.json", "w+") as f:
            json.dump(data, f)
        print(random.choice(exit_responses))
    else:
        print(random.choice(exit_responses))

# calls main function
if __name__ == '__main__':
    main()