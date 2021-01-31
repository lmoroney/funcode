# Packages to install: tensorflow, beautifulsoup4, wget
import tensorflow as tf
import numpy as np
import wget
import ssl
import os
import zipfile
import string
import pathlib
import json
from bs4 import BeautifulSoup


class TextClassificationHelper:
    # Public Class Variables
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    num_epochs = 30
    glove_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.twitter.27B.25d.zip"
    data_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
    glove_local = "glove.zip"
    data_local = "sarcasm.json"
    training_quotient = .8  # How much of the dataset will be used for training
    vocab_quotient = .5  # How much of the total vocab will be used

    # Private Class Variables
    __embedding_dim = 25  # Because GLove is 25 dimensions
    __training_labels = []
    __testing_labels = []
    __training_padded = []
    __testing_padded = []
    __embedding_matrix = []
    __glove_embeddings = dict()
    __saved_model_dir = export_dir = 'saved_model/1'
    __tflite_model_name = 'model.tflite'
    __vocab_size = 0  # The private vocab size that is calculated from the distinct # of words * the vocab quotient
    __tokenizer = None
    __max_length = None

    def create_model(self):
        # We'll start by downloading and preparing the glove embeddings into a dicitonary
        
        self.download_and_prepare_glove()
        self.download_and_process_data_file()
        self.train_model()

    def download_and_prepare_glove(self):
        # This function downloads the glove embeddings to a local zip
        # then extracts them to a .txt file from which it will read
        # the embeddings into a dictionary that it then returns to the caller
        ssl._create_default_https_context = ssl._create_unverified_context
        if os.path.exists(self.glove_local):
            print("Glove embeddings already downloaded...")
        else:
            print("Downloading Glove embeddings...")
            wget.download(self.glove_url, self.glove_local)

        if os.path.exists("glove/glove.twitter.27B.25d.txt"):
            print("Glove embeddings already extracted...")
        else:
            print("Unzipping Glove embeddings...")
            local_zip = self.glove_local
            zip_ref = zipfile.ZipFile(local_zip, 'r')
            zip_ref.extractall('glove')
            zip_ref.close()

        print("Creating dictionary from Glove embeddings...")
        f = open('glove/glove.twitter.27B.25d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.__glove_embeddings[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(self.__glove_embeddings))

    def download_and_process_data_file(self):
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
                     "as",
                     "at",
                     "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
                     "did",
                     "do",
                     "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
                     "have",
                     "having",
                     "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
                     "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
                     "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
                     "ought",
                     "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes",
                     "should",
                     "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves",
                     "then",
                     "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those",
                     "through",
                     "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve",
                     "were",
                     "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom",
                     "why",
                     "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
                     "yourselves"]

        table = str.maketrans('', '', string.punctuation)
        if os.path.exists(self.data_local):
            print("Data already downloaded...")
        else:
            print("Downloading sarcasm data file...")
            wget.download(self.data_url, self.data_local)

        print("Preprocessing JSON File...")
        import json
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        with open(self.data_local, 'r') as f:
            datastore = json.load(f)

        sentences = []
        labels = []
        urls = []
        for item in datastore:
            sentence = item['headline'].lower()
            sentence = sentence.replace(",", " , ")
            sentence = sentence.replace(".", " . ")
            sentence = sentence.replace("-", " - ")
            sentence = sentence.replace("/", " / ")
            soup = BeautifulSoup(sentence, features='html.parser')
            sentence = soup.get_text()
            words = sentence.split()
            filtered_sentence = ""
            for word in words:
                word = word.translate(table)
                if word not in stopwords:
                    filtered_sentence = filtered_sentence + word + " "
            sentences.append(filtered_sentence)
            labels.append(item['is_sarcastic'])
            urls.append(item['article_link'])

        print("Creating training and test datasets...")
        count, self.__max_length = self.get_distinct_word_count_and_length(sentences)
        self.__vocab_size = int(count * self.vocab_quotient)

        to_cut = int(len(sentences) * self.training_quotient)
        training_sentences = sentences[0:to_cut]
        testing_sentences = sentences[to_cut:]
        self.__training_labels = labels[0:to_cut]
        self.__testing_labels = labels[to_cut:]
        self.__tokenizer = Tokenizer(num_words=self.__vocab_size, oov_token=self.oov_tok)
        self.__tokenizer.fit_on_texts(training_sentences)

        word_index = self.__tokenizer.word_index
        print(len(word_index))
        training_sequences = self.__tokenizer.texts_to_sequences(training_sentences)
        self.__training_padded = pad_sequences(training_sequences, maxlen=self.__max_length, padding=self.padding_type,
                                               truncating=self.trunc_type)

        testing_sequences = self.__tokenizer.texts_to_sequences(testing_sentences)
        self.__testing_padded = pad_sequences(testing_sequences, maxlen=self.__max_length, padding=self.padding_type,
                                              truncating=self.trunc_type)

        xs = []
        ys = []
        cumulative_x = []
        cumulative_y = []
        total_y = 0
        for word, index in self.__tokenizer.word_index.items():
            xs.append(index)
            cumulative_x.append(index)
            if self.__glove_embeddings.get(word) is not None:
                total_y = total_y + 1
                ys.append(1)
            else:
                ys.append(0)
            cumulative_y.append(total_y / index)
        self.__embedding_matrix = np.zeros((self.__vocab_size, self.__embedding_dim))
        for word, index in self.__tokenizer.word_index.items():
            if index > self.__vocab_size - 1:
                break
            else:
                embedding_vector = self.__glove_embeddings.get(word)
                if embedding_vector is not None:
                    self.__embedding_matrix[index] = embedding_vector

        self.__training_padded = np.array(self.__training_padded)
        self.__training_labels = np.array(self.__training_labels)
        self.__testing_padded = np.array(self.__testing_padded)
        self.__testing_labels = np.array(self.__testing_labels)

    def train_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.__vocab_size, self.__embedding_dim, weights=[self.__embedding_matrix],
                                      trainable=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(self.__training_padded, self.__training_labels, epochs=self.num_epochs,
                            validation_data=(self.__testing_padded, self.__testing_labels), verbose=2)
        export_dir = self.__saved_model_dir
        tf.saved_model.save(model, export_dir)
        # Convert and save the model to tflite
        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        tflite_model = converter.convert()
        tflite_model_file = pathlib.Path(self.__tflite_model_name)
        tflite_model_file.write_bytes(tflite_model)
        # Write the dictionary and metadata for the model
        with open('word_dict.json', 'w') as file:
            json.dump(self.__tokenizer.word_index, file)

        meta_data = {"max_length": self.__max_length}
        with open('metadata.json', 'w') as file:
            json.dump(meta_data, file)

    def get_distinct_word_count_and_length(self, sentences):
        from collections import Counter
        corpus = ""
        sentence_lengths = 0
        for sentence in sentences:
            corpus = corpus + sentence
            sentence_lengths = sentence_lengths + len(sentence)
        foo = Counter(corpus.split())
        average_length = int(sentence_lengths / len(sentences))
        return len(foo.items()), average_length



