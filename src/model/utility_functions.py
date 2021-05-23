import re
import numpy as np
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


## Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']




def elongate_words(tweet):
    tweet = re.sub(r"he's | hes", "he is ", tweet)
    tweet = re.sub(r"there's | theres", "there is ", tweet)
    tweet = re.sub(r"we're", "we are ", tweet)
    tweet = re.sub(r"thats", "that is ", tweet)
    tweet = re.sub(r"won't | wont", "will not ", tweet)
    tweet = re.sub(r"they're | theyre", "they are ", tweet)
    tweet = re.sub(r"can't | cant", "cannot ", tweet)
    tweet = re.sub(r"wasn't | wasnt", "was not ", tweet)
    tweet = re.sub(r"don't | dont", "do not ", tweet)
    tweet = re.sub(r"aren't | arent", "are not ", tweet)
    tweet = re.sub(r"isn't | isnt", "is not ", tweet)
    tweet = re.sub(r"what's | whats", "what is ", tweet)
    tweet = re.sub(r"haven't | havent", "have not ", tweet)
    tweet = re.sub(r"hasn't | hasnt", "has not ", tweet)
    tweet = re.sub(r"there's | theres", "there is ", tweet)
    tweet = re.sub(r"it's | its", "it is ", tweet)
    tweet = re.sub(r"you're | youre", "you are ", tweet)
    tweet = re.sub(r"i'm | im", "i am ", tweet)
    tweet = re.sub(r"shouldn't | shouldnt", "should not ", tweet)
    tweet = re.sub(r"wouldn't | wouldnt", "would not ", tweet)
    tweet = re.sub(r"here's | heres", "here is ", tweet)
    tweet = re.sub(r"you've | youve", "you have ", tweet)
    tweet = re.sub(r"couldn't | couldnt", "could not ", tweet)
    tweet = re.sub(r"we've | weve", "we have ", tweet)
    tweet = re.sub(r"doesn't | doesnt", "does not ", tweet)
    tweet = re.sub(r"who's | whos", "who is ", tweet)
    tweet = re.sub(r"i've", "i have ", tweet)
    tweet = re.sub(r"y'all | yall", "you all ", tweet)
    tweet = re.sub(r"would've | wouldve", "would have ", tweet)
    tweet = re.sub(r"it'll", "it will ", tweet)
    tweet = re.sub(r"we'll", "we will ", tweet)
    tweet = re.sub(r"he'll", "he will ", tweet)
    tweet = re.sub(r"y'all", "you all ", tweet)
    tweet = re.sub(r"weren't | werent", "were not ", tweet)
    tweet = re.sub(r"they'll", "they will ", tweet)
    tweet = re.sub(r"they'd", "they would ", tweet)
    tweet = re.sub(r"they've | theyve", "they have ", tweet)
    tweet = re.sub(r"i'd", "i would ", tweet)
    tweet = re.sub(r"should've | shouldve", "should have ", tweet)
    tweet = re.sub(r"where's | wheres", "where is ", tweet)
    tweet = re.sub(r"we'd", "we would ", tweet)
    tweet = re.sub(r"i'll", "i will ", tweet)
    tweet = re.sub(r"let's | lets", "let us ", tweet)   
    tweet = re.sub(r"didn't | didnt", "did not ", tweet)
    tweet = re.sub(r"ain't | aint", "am not ", tweet)
    tweet = re.sub(r"you'll", "you will ", tweet)
    tweet = re.sub(r"you'd | youd", "you would ", tweet)
    tweet = re.sub(r"haven't | havent", "have not ", tweet)
    tweet = re.sub(r"could've | couldve", "could have ", tweet)  
    tweet = re.sub(r"some1", "someone ", tweet)
    tweet = re.sub(r"yrs", "years ", tweet)
    tweet = re.sub(r"hrs", "hours ", tweet)
    tweet = re.sub(r"2morow|2moro", "tomorrow ", tweet)
    tweet = re.sub(r"2day", "today ", tweet)
    tweet = re.sub(r"4got|4gotten", "forget ", tweet)
    tweet = re.sub(r"b-day|bday", "birthday ", tweet)
    tweet = re.sub(r"mother's", "mother ", tweet)
    tweet = re.sub(r"mom's", "mom ", tweet)
    tweet = re.sub(r"dad's", "dad ", tweet)
    tweet = re.sub(r"hahah|hahaha|hahahaha", "haha ", tweet)
    tweet = re.sub(r"lmao|lolz|rofl|lol", "laugh ", tweet)
    tweet = re.sub(r"thanx|thnx", "thanks ", tweet)
    
    return tweet




def preprocess_data(textdata):
    processedText = []
    
    # Create Lemmatizer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]* | (pic.twitter\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        # Lower Case conversion
        tweet = tweet.lower()
        tweet = elongate_words(tweet)
        
        # Remove all URLs
        tweet = re.sub(urlPattern," ",tweet)
        # Remove all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, " " + emojis[emoji])        
        # Remove @USERNAME.
        tweet = re.sub(userPattern," ", tweet)
        # Remove all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            if word not in stopwordlist:
                if len(word)>1:
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText




def model_evaluate(model, X_test, y_test):

    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ["Negative", "Positive"]
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_values = [value for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    
    labels = [f'{v1}\n\n{v2}\n({v3})' for v1, v2, v3 in zip(group_names, group_values, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.set(font_scale=2)
    plt.figure(figsize = (10,7))
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':18}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':18}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':20}, pad = 20)
    
    return y_pred