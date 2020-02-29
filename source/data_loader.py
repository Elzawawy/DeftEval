import pandas as pd 
import os
import spacy
from spacy.lang.en import English
from pathlib import Path
from . import task1_converter

class DeftCorpusLoader(object):
    """"""
    def __init__(self, deft_corpus_path):
        super().__init__()
        self.corpus_path = deft_corpus_path
        self._default_train_output_path = os.path.join(deft_corpus_path, "deft_files/converted_train")
        self._default_dev_output_path = os.path.join(deft_corpus_path, "deft_files/converted_dev")
        # Load English tokenizer, tagger, parser, NER and word vectors
        self._parser = English()

    def convert_to_classification_format(self, train_output_path = None, dev_output_path = None):

        if train_output_path == None or dev_output_path == None: 
            train_output_path = self._default_train_output_path
            dev_output_path = self._default_dev_output_path
            if not os.path.exists(train_output_path):
                os.mkdir(train_output_path)
            if not os.path.exists(dev_output_path):
                os.mkdir(dev_output_path)
        else: 
            if not os.path.exists(train_output_path) or not os.path.exists(dev_output_path):
                raise ValueError("Passed value for one or both of the output paths is not a valid existing path.")
            if not os.path.isdir(train_output_path) or not os.path.isdir(dev_output_path):
                raise NotADirectoryError("Passed value for one or both of the output paths is not a valid directory.")
        
        self.converted_train_path = train_output_path
        self.converted_dev_path = dev_output_path

        train_source_path = os.path.join(self.corpus_path, "deft_files/train")
        dev_source_path = os.path.join(self.corpus_path, "deft_files/dev")
        task1_converter.convert(Path(train_source_path), Path(train_output_path))
        task1_converter.convert(Path(dev_source_path), Path(dev_output_path))

    def load_classification_data(self, train_data_path = None, dev_data_path = None, preprocess= False, clean=False):

        if(train_data_path ==  None or dev_data_path == None):
            if os.path.exists(self._default_train_output_path) and os.path.exists(self._default_dev_output_path):
                train_data_path = self._default_train_output_path
                dev_data_path = self._default_dev_output_path
            else:
                self.convert_to_classification_format()
                train_data_path = self.converted_train_path
                dev_data_path = self.converted_dev_path

        train_deft_files = os.listdir(train_data_path)
        train_dataframe = pd.DataFrame([])
        for file in train_deft_files:
            dataframe = pd.read_csv(os.path.join(train_data_path, file), sep="\t", header = None)
            dataframe.columns = ["Sentence","HasDef"]
            train_dataframe = train_dataframe.append(dataframe, ignore_index=True)

        dev_deft_files = os.listdir(dev_data_path)
        dev_dataframe = pd.DataFrame([])
        for file in dev_deft_files:
            dataframe = pd.read_csv(os.path.join(dev_data_path, file), sep="\t", header = None)
            dataframe.columns = ["Sentence","HasDef"]
            dev_dataframe = dev_dataframe.append(dataframe, ignore_index=True)

        if(preprocess):
            self.preprocess_data(train_dataframe)
            self.preprocess_data(dev_dataframe)
            
        if(clean and preprocess):
            self.clean_data(train_dataframe)
            self.clean_data(dev_dataframe)
        elif(clean):
            raise ValueError("Can't set `clean` flag to true if `preprocess` flag hasn't been already set.")

        return (train_dataframe, dev_dataframe)

    def explore_data(self, dataframe, split):
        print("\nHead of ", split," Dataframe:\n==============================================================")
        print(dataframe.head())
        print("==============================================================")
        print("Number of instances of ",split ,"is",len(dataframe))
        print("==============================================================")
        print("Statistics of Sentences:\n==============================================================")
        print(dataframe.Sentence.describe())

    def preprocess_data(self, dataframe):
        nlp = spacy.load('en_core_web_sm')
        dataframe["Parsed"] = dataframe.Sentence.apply(self._spacy_preprocessor)

    def clean_data(self, dataframe):
        for index,parsed_list in enumerate(dataframe["Parsed"]):
            if len(parsed_list) < 5:
                dataframe.drop(index, inplace=True)
        # remove duplicate sentences from corpus
        dataframe.drop_duplicates(subset= "Sentence", inplace=True)

    def _spacy_preprocessor(self, sentence):
        # Creating our tokens object, which is used to create documents with linguistic annotations.
        tokens = self._parser(sentence)
        # Lemmatizing each token and converting each token into lowercase
        # Removing stop words, punctuations, spaces and non alphanumeric characters.
        tokens = [ token.lemma_.lower().strip() for token in tokens 
                    if not token.is_stop and not token.is_punct and 
                        not token.is_space and token.is_alpha ]
        # return preprocessed list of tokens
        return tokens
