from transformers import pipeline

def test_pipeline():
    try:
        text_generator = pipeline("text2text-generation")
        result = text_generator("Correct this text: I is a good boy.")
        print("Pipeline test successful. Result:", result)
    except Exception as e:
        print("Pipeline test failed. Error:", e)

if __name__ == "__main__":
    test_pipeline()
