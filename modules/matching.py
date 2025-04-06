import pandas as pd
import numpy as np


# This class checks the whether the given information is consistent before giving it to the model.

class InformationChecker:
    def __init__(self, df_passport, df_account, df_desc, df_profile, df_labels):
        self.df_passport = df_passport
        self.df_account = df_account
        self.df_desc = df_desc
        self.df_profile = df_profile
        self.df_labels = df_labels
        self.rejections = []

    # TODO: Understanding the complex matching
    def complex_matching(self): 
        pass

    def basic_matching(self):
        # Checking the passport and the profile
        for index, row in self.df_passport.iterrows():
            profile_row = self.df_profile.loc[self.df_profile['client_id'] == row['client_id']]
            profile_row = profile_row.iloc[0]


            # 1. Checking the name.
            profile_name = str(profile_row['name']).lower().replace(" ", "")
            first_name = str(row['first_name']).lower().replace(" ", "")
            middle_name = str(row['middle_name']).lower().replace(" ", "") if not pd.isna(row['middle_name']) else ""
            last_name = str(row['last_name']).lower().replace(" ", "")
            passport_name =  first_name + middle_name + last_name
            if profile_name != passport_name:
                print(f"Rejected client {row['client_id']} for unmatching name {profile_name} and {passport_name}")
                self.rejections.append(row['client_id'])
                continue

            # 2. Checking the birthdate
            if profile_row['birth_date'] != row['birth_date']:
                print(f"Rejected client {row['client_id']} for unmatching birthdate")
                self.rejections.append(row['client_id'])
                continue

            # 3. Checking the country
            if str(profile_row['nationality']).lower() != str(row['nationality']).lower():
                print(f"Rejected client {row['client_id']} for unmatching nationality")
                self.rejections.append(row['client_id'])
                continue
            
            # 4. Checking the passport number
            if profile_row['passport_number'] != row['passport_number']:
                print(f"Rejected client {row['client_id']} for unmatching passport number")
                self.rejections.append(row['client_id'])
                continue

            # 5. Other passport checks
            if profile_row['passport_issue_date'] != row['passport_issue_date'] or profile_row['passport_expiry_date'] != row['passport_expiry_date']:
                print(f"Rejected client {row['client_id']} for unmatching passport information")
                self.rejections.append(row['client_id'])
                continue
            
            # 6. Checking the gender
            if profile_row['gender'] != row['gender']:
                print(f"Rejected client {row['client_id']} for unmatching gender")
                self.rejections.append(row['client_id'])
                continue

        print(f'Number of rejections is {len(self.rejections)}')

    def create_mask(self):
        self.basic_matching()
        return self.rejections
    

    def save_rejections_to_csv(self, filename='rejections.csv'):
        if self.rejections:  # Only save if there are rejections
            df_rejections = pd.DataFrame({
                'client_id': self.rejections,
                'prediction': [0] * len(self.rejections)  # Assuming '0' means rejection
            })
            df_rejections.to_csv(filename, index=False)
            print(f"Rejections saved to {filename}")
        else:
            print("No rejections to save.")



df_passport = pd.read_csv('passport.csv')
df_account = pd.read_csv('account.csv')
df_desc = pd.read_csv('description.csv')
df_profile = pd.read_csv('profile.csv')
df_labels = pd.read_csv('labels.csv')
checker = InformationChecker(df_passport, df_account, df_desc, df_profile, df_labels)
checker.create_mask()
checker.save_rejections_to_csv()