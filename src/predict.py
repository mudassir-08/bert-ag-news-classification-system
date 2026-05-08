from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="../models/bert-ag-news"
)

def predict_news(text):
    result = classifier(text)
    return result

if __name__ == "__main__":
    sample = "Stock market crashes after inflation rises."
    prediction = predict_news(sample)
    print(prediction)