import spacy
import random
import en_core_web_lg
from spacy.util import minibatch, compounding
from pathlib import Path
from sklearn.metrics import classification_report

class DeftSpacyClassifier(object):

    def __init__(self, lang_model= None, positive_label="POSITIVE", negative_label="NEGATIVE"):
        super().__init__()
        if lang_model is not None: 
            self.nlp = spacy.load(lang_model)  # load existing spaCy model.
            print("Loaded model '%s'" %lang_model)
        else:
            self.nlp = en_core_web_lg.load()     # load default langauge model.
            print("Loaded default model '%s'" % en_core_web_lg)
        
        if "textcat" not in self.nlp.pipe_names:
            # Architecture: a neural network model where token vectors are calculated using a CNN. 
            self.textcat = self.nlp.create_pipe("textcat", 
                config={"exclusive_classes": True, "architecture": "simple_cnn"})
            self.nlp.add_pipe(self.textcat, last=True)
        else:
            self.textcat = self.nlp.get_pipe("textcat")

        self.classifier_output = None
        # add label to text classifier pipe.
        self.POSITIVE = positive_label
        self.NEGATIVE = negative_label
        self.textcat.add_label(self.POSITIVE)
        self.textcat.add_label(self.NEGATIVE)

    def fit(self, train_texts, dev_texts, train_cats, dev_cats, n_iter=20, loss_tol=0.005, output_dir = None):

        train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat"]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            # Initialize the pipe for training
            optimizer = self.nlp.begin_training()
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))

        # Yield an infinite series of compounding values.
        batch_sizes = compounding(32.0, 100.0, 1.001)
        # The Training Loop
        for i in range(n_iter):
            losses = {}
            random.shuffle(train_data)
            # Iterate over batches of items, batch-size vary on each step.
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                self.nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with self.textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = self.evaluate(self.nlp.tokenizer, dev_texts, dev_cats)
            print("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"]))
            # Early Stopping Condition.
            if(losses["textcat"] <= loss_tol):
              break
        if output_dir is not None:
            output_dir = Path(output_dir)
            # If it doesn't exist, no worries let's create it!
            if not output_dir.exists():
                output_dir.mkdir()
            with self.nlp.use_params(optimizer.averages):
                # Serialize the pipe to disk.
                self.nlp.to_disk(output_dir)
            print("Saved model to", output_dir)
            # set it for the score function to be computed later when needed.
            self.classifier_output = output_dir

    def evaluate(self, tokenizer, texts, cats):
        docs = (tokenizer(text) for text in texts)
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, doc in enumerate(self.textcat.pipe(docs)):
            gold = cats[i]
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if label == self.NEGATIVE:
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

    def score(self, dev_texts, dev_labels):
        if(self.classifier_output is None):
            raise NotImplementedError("You must save the model to an output directory after training, in order to use this function. Other ways are not implemented currently.")
        nlp = spacy.load(self.classifier_output)
        predicted = []
        for index,item in enumerate(dev_texts):
            doc = nlp(item)
            predict_doc = 1 if doc.cats[self.POSITIVE] > doc.cats[self.NEGATIVE] else 0
            predicted.append(predict_doc)
        # Print the resulting scores.
        print(classification_report(list(dev_labels), predicted))
