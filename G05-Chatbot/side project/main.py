#Importing all materials
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#Reading the intent file
with open("intents.json") as file:
    data = json.load(file)

try:  #Using some pickled data for our process if our data was processed successfully
    with open("data.pickle", "rb") as f:
        words, lables, training, output = pickle.load(f)  #Save these 4 variable in our pickle file and loading our list
except:  #Processing the data we need
    words = []
    lables = []
    docs_x = []
    docs_y = []

    #Looping through all the pattern in intent file
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  #Tokenizing or getting all the words
            words.extend(wrds)  #Put all the words in our words list
            docs_x.append(wrds)  #Add our tokenized words into docs list
            docs_y.append(intent["tag"])  #Saving the tag of our pattern

        if intent["tag"] not in lables:
            lables.append(intent["tag"]) #Getting all the tags and put in lables list

    #Making a bag of words to get ready to put it in our model
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  #If statement makes it to not care about question mark
    words = sorted(list(set(words)))  #Making sure that there are no double words in list

    lables = sorted(lables)  #Sorting our lables list

    training = []
    output = []

    out_empty = [0 for _ in range(len(lables))]  #Creating an empty output for putting the words we need to fill it with

    #Creating bag of words for our output
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:  #If the word exist in pattern we looping through
                bag.append(1) #Showing if that word existes and where it is
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[lables.index(docs_y[x])] = 1  #Seeing where the tag we want is and set in our output

        training.append(bag)
        output.append(output_row)

    #Changing the list to array
    training = numpy.array(training) 
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, lables, training, output), f)  #Writting all 4 variables into our pickle file so we can save it

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])  #Input data #Finding the input shape that we are expecting for our model
net = tflearn.fully_connected(net, 8)  #Hidden layer #Adding fully connected layer to our network
net = tflearn.fully_connected(net, 8)  #Hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  #Output layer #Getting the probability of each output
net = tflearn.regression(net)

model = tflearn.DNN(net)  #DNN will get the network we made and will use it

#If the model already existes, it won't train the model again
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #Fitting the model or passing all our data #n_epoch is the amount of time it's going to see the data 
    model.save("model.tflearn")  #Saving our model

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]  #Create a blank bag of words list
    
    s_words = nltk.word_tokenize(s)  #Getting a list of tokenized words
    s_words = [stemmer.stem(words.lower()) for words in s_words]  #Stemming all our words

    #Generating our bag of words
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:   #If the current word we are looking at in our list, is the same as the word in our sentence
                bag[i] = 1  #Putting 1 in list means that word exists

    return numpy.array(bag)  #Getting the bag of words list and turn it into array

def chat():
    print("Start talking with the bot! (Type 'quit' to stop)")
    while True:
        inp = input("You: ")  #Getting the user's response
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)  #Giving the greatest index number 
        tag = lables[results_index]  #Giving the lable that think the message is
        
        #Getting the responses we need
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))

        else:
            print("I didn't get that, Ask something else!")
chat()       