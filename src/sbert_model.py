from sentence_transformers import SentenceTransformer, util
import re
from transformers import pipeline

NEGATIONS = {"not", "no", "never", "n't"}

OPPOSITES = {
    ("boy", "girl"),
    ("man", "woman"),
    ("male", "female"),
    ("yes", "no"),
    ("true", "false")
}

def has_opposite_words(s1, s2):
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())

    for w1, w2 in OPPOSITES:
        if (w1 in words1 and w2 in words2) or (w2 in words1 and w1 in words2):
            return True
    return False


def has_negation(sentence):
    words = set(re.findall(r"\b\w+\b", sentence.lower()))
    return any(n in words for n in NEGATIONS)

class SBERTModel:
    def __init__(self, model_choice='MiniLM'):
        self.model_choice = model_choice
        self.is_cross_encoder = False
        
        if model_choice == 'MiniLM':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif model_choice == 'MPNet':
            self.model = SentenceTransformer('all-mpnet-base-v2')
        elif model_choice == 'Elite':
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder('cross-encoder/stsb-roberta-large')
            self.is_cross_encoder = True
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

    def is_contradiction(self, s1, s2):
        result = self.nli(f"{s1} </s> {s2}")[0]
        return result['label'] == 'CONTRADICTION'

    def similarity(self, s1, s2):
        if self.is_cross_encoder:
            score = float(self.model.predict([s1, s2]))
        else:
            embeddings = self.model.encode([s1, s2], convert_to_tensor=True)
            score = util.cos_sim(embeddings[0], embeddings[1]).item()

        # Negation penalty
        if has_negation(s1) != has_negation(s2):
            score *= 0.5   # reduce similarity

        # NLI contradiction check
        if self.is_contradiction(s1, s2):
            score = 0.0

        # Opposite word penalty
        if has_opposite_words(s1, s2):
            score *= 0.3

        return score

    def predict(self, dataset, threshold=0.75):
        preds = []
        for s1, s2 in zip(dataset["sentence1"], dataset["sentence2"]):
            score = self.similarity(s1, s2)
            preds.append(1 if score > threshold else 0)
        return preds