from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# define some variables
number_of_words_in_vocabulary = 88000
num_samples_val = 5000


def create_model():
    model = Sequential()
    model.add(Embedding(number_of_words_in_vocabulary, 16))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()

    return model


def decode_review(text, word_index):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded = " ".join([reverse_word_index.get(i, "?") for i in text])

    return decoded


def encode_review(text, word_index):
    encoded = [1]  # start of line
    for word in text:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)  # unknown word

    return encoded


def create_word_index(data):
    word_index = data.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    return word_index


def get_data(data):
    (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=number_of_words_in_vocabulary)

    return train_data, train_labels, test_data, test_labels


def preprocessing(train_data, test_data, word_index):
    train_preprocess = pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
    test_preprocess = pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

    return train_preprocess, test_preprocess


def split_data(train_data, train_labels, num_samples_val=5000):
    x_val = train_data[:num_samples_val]
    y_val = train_labels[:num_samples_val]
    x_train = train_data[num_samples_val:]
    y_train = train_labels[num_samples_val:]

    return x_train, y_train, x_val, y_val
