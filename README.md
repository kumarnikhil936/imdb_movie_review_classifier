# imdb_movie_review_classifier
Performed sentiment analysis and text classification on the IMDB dataset.

More information about the dataset: 

https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset

This is the IMDB dataset that contains the movie reviews. This dataset contains movie reviews along with their associated binary sentiment polarity labels. It is intended to serve as a benchmark for sentiment classification. The core dataset contains 50,000 reviews split evenly into 25k train and 25k test sets. The overall distribution of labels is balanced (25k pos and 25k neg). 

Each example is a sentence representing the movie review and a corresponding label. The sentence is not preprocessed in any way. The label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

Model:

Embedding layer is used to represent text vectors and later a global average pooling layer has been used to reduce the dimensionality. Further dense layers are stacked to make the complete neural network acrhitecture. 

The utils.py file contains the required helper functions for the operations like creation of the model, word index dictionaries, encoding and decoding strings, etc. 

The main.py file has the code for the actual training of neural network and performing inference on the IMDB dataset. 

The test.py file loads a sample review present in the review.txt file and performs inference to see if the sentiment in this review was positive or negative. 

The savedModel.h5 file is a pretrained model and can be directly used to perform classification tasks. 


References:

Potts, Christopher. 2011. On the negativity of negation. In Nan Li and David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20, 636-659.
