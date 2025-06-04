import os
os.environ["USE_TF"]= "1"
os.environ["TRANSFORMERS_NO_TORCH"] = "1"
os.environ["TRANSFORMERS_BACKEND"] = "tensorflow"
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

class BERTProcessor:
    def __init__(self):
        """
        Initialize the BERTProcessor with a pre-trained BERT base uncased model for semantic understanding.
        Uses only TensorFlow backend.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = TFAutoModel.from_pretrained("bert-base-uncased")

    def get_embeddings(self, text: str) -> tf.Tensor:
        """
        Generate BERT embeddings for the input text.

        Args:
            text: The input text to process.

        Returns:
            Tensor containing the BERT embeddings.
        """
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between embeddings of two texts.

        Args:
            text1: First text input.
            text2: Second text input.

        Returns:
            Cosine similarity score between the two texts.
        """
        embeddings1 = tf.reduce_mean(self.get_embeddings(text1), axis=1)
        embeddings2 = tf.reduce_mean(self.get_embeddings(text2), axis=1)
        similarity = tf.keras.losses.cosine_similarity(embeddings1, embeddings2)
        return -similarity.numpy()[0]  # Convert to positive similarity score
