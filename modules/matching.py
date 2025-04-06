import pandas as pd
import numpy as np
import ast


# This class checks the whether the given information is consistent before giving it to the model.

class InformationChecker:
    def __init__(self, df_passport, df_account, df_desc, df_profile, df_labels, test_mode=False):
        self.df_passport = df_passport
        self.df_account = df_account
        self.df_desc = df_desc
        self.df_profile = df_profile
        self.df_labels = df_labels
        self.test_mode = test_mode
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

        # Checking the account and the profile
        for index, account_row in self.df_account.iterrows():
            # Retrieve the corresponding profile entry
            profile_row = self.df_profile.loc[self.df_profile['client_id'] == account_row['client_id']]
            if profile_row.empty:
                print(f"No profile found for client {account_row['client_id']}")
                continue
            profile_row = profile_row.iloc[0]
            # Fields to check for consistency
            fields_to_check = [
                'currency', 'address_city', 'address_street_name', 'address_street_number',
                'address_postal_code', 'phone_number', 'email_address'
            ]
            
            for field in fields_to_check:
                if str(account_row[field]).strip().lower() != str(profile_row[field]).strip().lower():
                    print(f'Rejected client {profile_row['client_id']}for {field}')
                    self.rejections.append(profile_row['client_id'])

        # Checking the account and passport
        for index, account_row in self.df_account.iterrows():
            passport_row = self.df_passport.loc[self.df_passport['client_id'] == account_row['client_id']]
            passport_row = passport_row.iloc[0]
            fields_to_check = [
                 "first_name", "middle_name", "last_name", "passport_number", "passport_number",
            ]
            for field in fields_to_check:
                if str(account_row[field]).strip().lower() != str(passport_row[field]).strip().lower():
                    print(f'Rejected client {passport_row['client_id']}for {field}')
                    self.rejections.append(passport_row['client_id'])

        # Checking the preffered_markets 
        # merged_df = self.df_profile.merge(self.df_passport, on='client_id').merge(self.df_account, on='client_id')

        # # Step 2: Convert 'preferred_markets' column to actual lists
        # merged_df['preferred_markets'] = merged_df['preferred_markets'].apply(ast.literal_eval)
        # # print(merged_df.columns)
        # # print('***')
        # # return
        # for index, row in merged_df.iterrows():
        #     options = []
        #     options.append(row['country'])
        #     options.append(row['country_of_domicile_x'])
        #     for market in row['preferred_markets']:
        #         if market not in options:
        #             print(f'Rejected client {row['client_id']}for not amtching infooo')
        #             self.rejections.append(row['client_id'])

        print(f'Number of rejections is {len(self.rejections)}')
        
        if not self.test_mode: 
            for index, row in self.df_labels.iterrows():
                if row['label'] == 1 and row['client_id'] in self.rejections:
                    print(f"{row['client_id']} is wrong")
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