import pandas as pd
import numpy as np


# This class checks the whether the given information is consistent before giving it to the model.

class InformationChecker:
    def __init__(self, df_passport, df_account, df_desc, df_profile):
        self.df_passport = df_passport
        self.df_account = df_account
        self.df_desc = df_desc
        self.df_profile = df_profile
        self.rejections = []

    # TODO: Understanding the complex matching
    def complex_matching(self): 
        pass

    def basic_matching(self):
        # Checking the passport and the profile
        for index, row in self.df_passport.iterrows():
            profile_row = self.df_profile.loc[self.df_profile['client_id'] == row['client_id']]
            profile_row = profile_row.iloc[0]
            profile_name = str(profile_row['name']).lower().replace(" ", "")
            passport_name = str(row['first_name']).lower() + str(row['middle_name']).lower() + str(row['last_name']).lower()
            if profile_name != passport_name:
                print(f'Rejected client {row['client_id']} for unmatching name')
                self.rejections.append(row['client_id'])
        print(f'Number of rejections is {len(self.rejections)}')
    def create_mask(self):
        pass