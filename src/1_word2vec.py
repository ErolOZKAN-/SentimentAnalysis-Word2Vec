import sys
import nltk.data,logging, re
import numpy as np
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf8')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

def reviewToSentences(review, tokenizer, remove_stopwords=False):
    review = unicode(review, errors='ignore')
    rawSentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in rawSentences:
        if len(raw_sentence) > 0:
            sentences.append(reviewToWordList(raw_sentence, remove_stopwords))
    return sentences

def reviewToWordList(review, remove_stopwords=False):
    reviewText = BeautifulSoup(review).get_text() # Remove HTML
    reviewText = re.sub("[^a-zA-Z]"," ", reviewText)    # Remove non-letters
    words = reviewText.lower().split()    # Convert words to lower case and split them

    # 4. Optionally remove stop words (
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

# Function to average all of the word vectors in a given paragraph
def makeFeatureVec(words, model, num_features):

    # initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

# Given a set of reviews (each one a list of words), calculate the average feature vector for each one and return a 2D numpy array
def getAvgFeatureVecs(reviews, model, num_features):
    # Initialize a counter
    counter = 0.

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    # Loop through the reviews
    for review in reviews:

        if counter%1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))

        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
        counter = counter + 1.

    return reviewFeatureVecs

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(reviewToWordList(review, remove_stopwords=True))
    return clean_reviews

if __name__ == '__main__':
    # Read data from files
    train = pd.read_csv( '../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv( '../data/testData.tsv', header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( "../data/unlabeledTrainData.tsv", header=0,  delimiter="\t", quoting=3 )

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size )

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += reviewToSentences(review, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences += reviewToSentences(review, tokenizer)

    # Set parameters and train the word2vec model
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model=Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling,seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    #SAVE THE MODEL
    model_name = "../data/300features_40minwords_10context"
    model.save(model_name)

    # TEST THE MODEL
    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")

    # Create average vectors for the training and test sets
    print "Creating average feature vecs for training reviews"
    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )

    print "Creating average feature vecs for test reviews"
    testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )

    # Fit a random forest to the training set, then make predictions
    forest = RandomForestClassifier( n_estimators = 100 )

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit( trainDataVecs, train["sentiment"] )

    # Test & extract results
    result = forest.predict( testDataVecs )

    # Write the test results
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "../data/OUTPUT_Word2Vec_AverageVectors.csv", index=False, quoting=3 )
    print "Wrote Word2Vec_AverageVectors.csv"