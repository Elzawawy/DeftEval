import pandas as pd 
import os
from scripts import task1_converter 
from pathlib import Path

class DeftCorpusLoader(object):
    """"""
    def __init__(self, deft_corpus_path):
        super().__init__()
        self.corpus_path = deft_corpus_path
        self._default_train_output_path = "deft_files/converted_train"
        self._default_dev_output_path = "deft_files/converted_dev"

    def convert_to_classification_format(self, train_output_path = None, dev_output_path = None):

        if train_output_path == None or dev_output_path == None: 
            train_output_path = os.path.join(self.corpus_path, self._default_train_output_path)
            dev_output_path = os.path.join(self.corpus_path, self._default_dev_output_path)
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

    def load_classification_data(self, train_data_path = None, dev_data_path = None):
        
        if(train_data_path ==  None or dev_data_path == None):
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
        
        return (train_dataframe, dev_dataframe)

