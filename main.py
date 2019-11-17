from tensorflow.keras.datasets import imdb

from utils import create_model, decode_review, get_data, preprocessing, create_word_index, split_data

if __name__ == '__main__':
    # Fetch the dataset
    data = imdb
    word_index = create_word_index(data)

    # Create training, testing and validation splits
    train_data, train_labels, test_data, test_labels = get_data(data)
    train_data, test_data = preprocessing(train_data, test_data, word_index)
    x_train, y_train, x_val, y_val = split_data(train_data, train_labels)

    # Create, compile, train, and evaluate the model
    model = create_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    fitModel = model.fit(x_train, y_train, epochs=1000, batch_size=512, validation_data=(x_val, y_val), verbose=1)
    results = model.evaluate(test_data, test_labels)

    print("Accuracy: %.2f" % results[1] * 100)

    # Save the trained model
    model.save("savedModel.h5")

    # Test performance on a sample review
    sample_review = test_data[15]
    predict = model.predict([sample_review])

    print("\nEncoded Review:\n", str(sample_review))
    print("\nDecoded Review:\n", str(decode_review(sample_review, word_index)))
    print("\nPrediction: ", str(predict[0]), "\nActual: ", str(test_labels[15]))
