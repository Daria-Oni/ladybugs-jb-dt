import pandas as pd
import numpy as np

# Load your merged DataFrame
df = pd.read_csv("noduplicates.csv")

# Function to apply
def check_education(entry):
    if pd.notna(entry) and entry != '':
        return 1
    else:
        return 0

# Applying the function to the 'higher_education' column
df['higher_education'] = df['higher_education'].apply(check_education)

columns_to_drop = [
"name", "first_name", "middle_name", "last_name", "passport_number", "address_street_name", "address_street_number", "address_postal_code",
"phone_number", "email_address", "address", "client_id", "passport_issue_date", "passport_expiry_date", "secondary_school", "employment_history",
"aum", "inheritance_details", "real_estate_details", "secondary_school_name", "secondary_school_graduation_year", "Summary Note",
"Occupation History", "country_code", "passport_mrz","prediction"
]
# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True)



# Save the cleaned DataFrame
df.to_csv("finaldata.csv", index=False)

