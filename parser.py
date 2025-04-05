# The script for parsing the zip files.

import zipfile
import os

import json
import pandas as pd
import numpy as np

# Environment Variables
current_directory = os.getcwd()
print(current_directory)

# Unzipping the client files and creating the csv
extract_to = f'{current_directory}/data'
os.makedirs(extract_to, exist_ok=True)

for i in range(1,5):
    zip_path = f'{current_directory}/datathon_part{i}.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

for i in range(10000):
    zip_path = f'{extract_to}/client_{i}.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(f'{extract_to}/client_{i}')
    os.remove(zip_path)
    print(f"Deleted: {zip_path}")

data_path = os.path.join(current_directory, 'data')

account_columns = [
    "name",
    "first_name",
    "middle_name",
    "last_name",
    "passport_number",
    "currency",
    "address_city",
    "address_street_name",
    "address_street_number",
    "address_postal_code",
    "country_of_domicile",
    "phone_number",
    "email_address"
]

description_columns = ["Summary Note", "Family Background", "Education Background", "Occupation History", "Wealth Summary", "Client Summary"]

profile_columns = [
    "name",
    "address_city",
    "address_street_name",
    "address_street_number",
    "address_postal_code",
    "country_of_domicile",
    "birth_date",
    "nationality",
    "passport_number",
    "passport_issue_date",
    "passport_expiry_date",
    "gender",
    "phone_number",
    "email_address",
    "marital_status",
    "secondary_school",
    "higher_education",
    "employment_history",
    "aum",
    "inheritance_details",
    "real_estate_details",
    "investment_risk_profile",
    "investment_horizon",
    "investment_experience",
    "type_of_mandate",
    "preferred_markets",
    "currency",
    'secondary_school_name',
    'secondary_school_graduation_year',
    'aum_savings',
    'aum_inheritance',
    'aum_real_estate_value',
    'inheritance_details_relationship',
    'inheritance_details_inheritance_year',
    'inheritance_details_profession'
]

passport_columns = [
    "first_name",
    "middle_name",
    "last_name",
    "gender",
    "country",
    "country_code",
    "nationality",
    "birth_date",
    "passport_number",
    "passport_mrz",  # list of strings
    "passport_issue_date",
    "passport_expiry_date",
]


df_account = pd.DataFrame(columns=account_columns)
df_description = pd.DataFrame(columns=description_columns)
df_profile = pd.DataFrame(columns=profile_columns)
df_passport = pd.DataFrame(columns=passport_columns)

for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith("client_"):
        client_id = folder_name.split('_')[-1]
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if 'address' in data:
                             data['address_city'] = data['address']['city']
                             data['address_street_name'] = data['address']['street name']
                             data['address_street_number'] = data['address']['street number']
                             data['address_postal_code'] = data['address']['postal code']
                        data['client_id'] = client_id
                        if file_name == 'account_form.json':
                            df_account = pd.concat([df_account, pd.DataFrame([data])], ignore_index=True)
                        elif file_name == 'passport.json':
                            print(f'Passport for client {client_id}')
                            df_passport = pd.concat([df_passport, pd.DataFrame([data])], ignore_index=True)
                        elif file_name == 'client_description.json':
                            print(f'Description for client {client_id}')
                            df_description = pd.concat([df_description, pd.DataFrame([data])], ignore_index=True)
                        elif file_name == 'client_profile.json':
                            for key, value in data['secondary_school'].items():
                                data[f'secondary_school_{key}'] = value
                            for key, value in data['aum'].items():
                                data[f'aum_{key}'] = value
                            for key, value in data['inheritance_details'].items():
                                data[f'inheritance_details_{key}'] = value
                            print(f'Profile for client {client_id}')
                            df_profile = pd.concat([df_profile, pd.DataFrame([data])], ignore_index=True)

df_account.to_csv('account.csv', index=False)
df_profile.to_csv('profile.csv', index=False)
df_description.to_csv('description.csv', index=False)
df_passport.to_csv('passport.csv', index=False)