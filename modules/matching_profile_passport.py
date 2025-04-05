import pandas as pd
import numpy as np

class BasicDataChecker:
    def __init__(self, df_passport, df_profile):
        self.df_passport = df_passport
        self.df_profile = df_profile
        self.rejections = []

    def check_data_consistency(self):
        # Iterate through each passport record
        for index, passport_row in self.df_passport.iterrows():
            # Retrieve the corresponding profile entry
            profile_row = self.df_profile.loc[self.df_profile['client_id'] == passport_row['client_id']]
            if profile_row.empty:
                print(f"No profile found for client {passport_row['client_id']}")
                continue
            profile_row = profile_row.iloc[0]

            # Check concatenated names (first_name + middle_name + last_name)
            passport_full_name = (passport_row['first_name'].strip() + ' ' +
                                  (passport_row['middle_name'].strip() if not pd.isna(passport_row['middle_name']) else '') + ' ' +
                                  passport_row['last_name'].strip()).lower().replace(' ', '')
            profile_full_name = profile_row['name'].lower().replace(' ', '')
            if passport_full_name != profile_full_name:
                self.log_rejection(passport_row['client_id'], 'full name mismatch')

            # Check birth date
            if profile_row['birth_date'] != passport_row['birth_date']:
                self.log_rejection(passport_row['client_id'], 'birth date mismatch')

            # Check nationality
            if profile_row['nationality'].lower() != passport_row['nationality'].lower():
                self.log_rejection(passport_row['client_id'], 'nationality mismatch')

            # Check passport number
            if profile_row['passport_number'] != passport_row['passport_number']:
                self.log_rejection(passport_row['client_id'], 'passport number mismatch')

            # Check gender
            if profile_row['gender'] != passport_row['gender']:
                self.log_rejection(passport_row['client_id'], 'gender mismatch')

            # Check passport issue date
            if profile_row['passport_issue_date'] != passport_row['passport_issue_date']:
                self.log_rejection(passport_row['client_id'], 'passport issue date mismatch')

            # Check passport expiry date
            if profile_row['passport_expiry_date'] != passport_row['passport_expiry_date']:
                self.log_rejection(passport_row['client_id'], 'passport expiry date mismatch')

        # Output the results
        self.output_results()

    def log_rejection(self, client_id, reason):
        print(f'Rejected client {client_id} for {reason}')
        self.rejections.append(client_id)

    def output_results(self):
        print(f'Total rejections: {len(self.rejections)}')
        if self.rejections:
            print(f'Rejected client IDs: {self.rejections}')
            # Create a DataFrame for rejected results
            rejected_df = pd.DataFrame({
                'client_id': self.rejections,
                'prediction': [0] * len(self.rejections)  # Changed 'rejected' to 0 here
            })
            # Save to CSV
            rejected_df.to_csv('rejected_clients.csv', index=False)
            print('Rejection results saved to rejected_clients.csv')


df_passport = pd.read_csv('passport.csv')
df_profile = pd.read_csv('profile.csv')
checker = BasicDataChecker(df_passport, df_profile)
checker.check_data_consistency()