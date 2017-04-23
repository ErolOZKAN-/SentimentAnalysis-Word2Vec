from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

#READ DATA
train = pd.read_csv( '../data/labeledTrainData.tsv', header=0,       delimiter="\t", quoting=3)
test = pd.read_csv( '../data/testData.tsv', header=0, delimiter="\t",               quoting=3 )
print "COLUMN VALUES : ", train.columns.values
print 'The first review is:', train["review"][0]

def prepocess( raw_review ):
    reviewText = BeautifulSoup(raw_review).get_text() # 1. REMOVE HTML
    lettersOnly = re.sub("[^a-zA-Z]", " ", reviewText) # 2. REMOVE NON-LETTERS
    words = lettersOnly.lower().split() # CONVERT TO LOWERCASE AND SPLIT
    stops = set(stopwords.words("english")) # searching a set is much faster
    meaningfulWords = [w for w in words if not w in stops] # Remove stop words
    return( " ".join( meaningfulWords )) # Join the words back into one string separated by space,

# Get the number of reviews based on the dataframe column size
numberOfTrainReviews = train["review"].size
print "Cleaning and parsing the training data...\n"
cleanedTrainReviews = []
for i in xrange( 0, numberOfTrainReviews ):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, numberOfTrainReviews )
    cleanedTrainReviews.append( prepocess( train["review"][i] ))


print "Creating the bag of words model...\n"

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
trainDataFeatures = vectorizer.fit_transform(cleanedTrainReviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
trainDataFeatures = trainDataFeatures.toarray()
print trainDataFeatures.shape

vocab = vectorizer.get_feature_names()
print vocab

# Sum up the counts of each vocabulary word
dist = np.sum(trainDataFeatures, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
forest = forest.fit(trainDataFeatures, train["sentiment"])

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
numberOfTestReviews = len(test["review"])
cleanedTestReviews = []

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,numberOfTestReviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, numberOfTestReviews)
    cleanReview = prepocess(test["review"][i])
    cleanedTestReviews.append(cleanReview)

# Get a bag of words for the test set, and convert to a numpy array
testDataFeatures = vectorizer.transform(cleanedTestReviews)
testDataFeatures = testDataFeatures.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(testDataFeatures)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "../data/OUTPUT_Bag_of_Words_model.csv", index=False, quoting=3 )