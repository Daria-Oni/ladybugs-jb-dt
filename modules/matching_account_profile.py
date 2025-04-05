import pandas as pd
import numpy as np

class AccountProfileChecker:
    def __init__(self, df_account, df_profile):
        self.df_account = df_account
        self.df_profile = df_profile
        self.rejections = []

    def check_account_consistency(self):
        # Iterate through each account record
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
                    self.log_rejection(account_row['client_id'], f'{field} mismatch')

        # Output the results
        self.output_results()

    def log_rejection(self, client_id, reason):
        print(f'Rejected client {client_id} for {reason}')
        self.rejections.append(client_id)

    def output_results(self):
        print(f'Total rejections: {len(self.rejections)}')
        if self.rejections:
            print(f'Rejected client IDs: {self.rejections}')


df_account = pd.read_csv('account.csv')
df_profile = pd.read_csv('profile.csv')
checker = AccountProfileChecker(df_account, df_profile)
checker.check_account_consistency()