from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import encode_review, create_word_index

# prepare the word index dictionary
data = imdb
word_index = create_word_index(data)

# load the trained model
model = load_model("savedModel.h5")

# open the review file and perform inference
review_filename = "review.txt"
with open(review_filename, encoding="utf-8") as f:
    for line in f.readlines():
        # remove unwanted characters
        formattedLine = line.replace(",", "").replace(".", "").replace(":", "").replace("(", "").replace(")",
                                                                                                         "").replace(
            "\"", "").strip()
        # encode the line in integers
        encodedLine = encode_review(formattedLine, word_index)
        # pad the line if length is shorter than 250
        paddedLine = pad_sequences([encodedLine], value=word_index["<PAD>"], padding="post", maxlen=250)
        # perform prediction
        predict = model.predict(paddedLine)

        print("Original line:\n", line)
        print("Encoded line:\n", paddedLine)

        print("Prediction: %.2f" % predict[0])
