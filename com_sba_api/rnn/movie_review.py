import os
from com_sba_api.utils.file_helper import FileReader
from pandas import read_table
import numpy as np

class MovieReview:
    
    def __init__(self):
        self.reader = FileReader()
    
    def hook(self):
        self.load_corpus()    
        
    def load_corpus(self):
        reader = self.reader
        corpus = read_table('./data/movie_review.csv', sep = ',', encoding = 'utf-8')
        print(f'Corpus Spec : {corpus}')
        return np.array(corpus)
    
mr = MovieReview()
mr.hook()