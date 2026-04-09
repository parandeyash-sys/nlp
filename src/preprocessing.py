import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

def preprocess_dataset(dataset):
    for split in dataset:
        dataset[split] = dataset[split].map(
            lambda x: {
                "sentence1": clean_text(x["sentence1"]),
                "sentence2": clean_text(x["sentence2"])
            }
        )
    return dataset