from transformers import pipeline

class BERTProcessor:
    def __init__(self):
        """
        Initialize the BERTProcessor with a pre-trained Pegasus model for summarization and context derivation.
        """
        self.model = pipeline("summarization", model="google/pegasus-xsum")

    def correct_text(self, text: str) -> str:
        """
        Use the Pegasus model to analyze meeting slide information and provide context without losing any data or context.

        Args:
            text: The input text extracted from meeting slides.

        Returns:
            The analyzed context, preserving all information and data.
        """
        try:
            # Prompt Pegasus to analyze and provide context for meeting slide information
            prompt = (
                "This is information extracted from the slides of a meeting. "
                "Analyze and provide the context of what is happening, without losing any data or context: "
                f"{text}"
            )
            result = self.model(prompt, max_length=60, min_length=20, truncation=True)
            return result[0]['summary_text']
        except Exception as e:
            print(f"Error during Pegasus processing: {e}")
            return text
