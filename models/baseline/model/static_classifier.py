import re
import random


class StaticClassifier:
    """
    Basic classifier that utilizes knowledge from scraped open source data
    """
    def __init__(self, knowledge, slot_labels):
        self.knowledge = knowledge
        self.slot_labels = slot_labels

    def _infer_label_type(self, word):
        candidates = []
        for label, database in self.knowledge.items():
            # Match by vocabulary
            if 'text' in database:
                if word in database['text']:
                    candidates.append(label)
            # Match by pattern(s)
            else:
                patterns = [re.compile(e) for e in database['regex']]
                if any(regex.match(word) for regex in patterns):
                    candidates.append(label)
        # Return default 'O' for no founded candidate labels
        if not candidates:
            return 'O'
        # Return only candidate
        elif len(candidates) < 2:
            return candidates[0]
        # Randomly select a candidate in case multiple were found
        else:
            return random.choice(candidates)

    def predict(self, sentence):
        pred = []
        for w in sentence.split(' '):
            pred.append(self._infer_label_type(w))
        return pred

