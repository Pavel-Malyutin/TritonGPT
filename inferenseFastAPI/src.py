import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


class TextGenerator:
    def __init__(self, top_k=10):
        self.model_dir = self.__choose_last_version()
        self.k = top_k
        self.model = tf.keras.models.load_model(f'storage/{self.model_dir}/model')
        self.max_tokens = self.model.get_config()["layers"][0]["config"]["batch_input_shape"][-1]
        self.vocab = self.__get_vocab()
        self.word_to_index = self.get_word_to_index()

    @staticmethod
    def __choose_last_version() -> str:
        folder = list(sorted(os.listdir("storage"), reverse=True))[0]
        return folder

    def __get_vocab(self) -> dict:
        with open(f"storage/{self.model_dir}/vocab.json", "r") as f:
            vocab = json.load(f)
        return vocab

    def get_word_to_index(self) -> dict:
        word_to_index = {}
        for index, word in enumerate(self.vocab):
            word_to_index[word] = index
        return word_to_index

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.vocab[number]

    def generate(self, start_prompt: str) -> str:
        # todo
        start_tokens = [self.word_to_index.get(_, 1) for _ in start_prompt.split()]
        start_tokens = [_ for _ in start_tokens]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.max_tokens - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.max_tokens]
                sample_index = self.max_tokens - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in start_tokens + tokens_generated]
        )
        return txt
