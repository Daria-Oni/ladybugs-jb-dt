import pandas as pd
import numpy as np


# This class checks the whether the given information is consistent before giving it to the model.

class InformationChecker:
    def __init__(self, df_passport, df_account, df_desc, df_profile):
        self.df_passport = df_passport
        self.df_account = df_account
        self.df_desc = df_desc
        self.df_profile = df_profile

    # TODO: Understanding the complex matching
    def complex_matching(self): 
        pass

    def basic_matching(self):
        pass

    def create_mask(self):
        pass