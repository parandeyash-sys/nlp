from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression()

    def prepare_data(self, dataset):
        return [
            s1 + " " + s2
            for s1, s2 in zip(dataset["sentence1"], dataset["sentence2"])
        ]

    def train(self, train_data):
        X = self.vectorizer.fit_transform(self.prepare_data(train_data))
        y = train_data["label"]
        self.model.fit(X, y)

    def predict(self, test_data):
        X = self.vectorizer.transform(self.prepare_data(test_data))
        return self.model.predict(X)