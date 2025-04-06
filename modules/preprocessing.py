import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from modules.model import HybridModel
from sklearn.metrics import accuracy_score


# Function to calculate education level
def calculate_education_level(row):
    # Check if secondary_school is not NaN, and add 1 if true
    secondary_school_count = 1 if pd.notna(row['secondary_school']) else 0
    # Count the number of entries in higher_education (length of the list)
    higher_education_count = len(row['higher_education']) if isinstance(row['higher_education'], list) else 0
    # Total education level
    return secondary_school_count + higher_education_count

# Function to calculate the difference between the lowest start year and biggest end year
def calculate_year_diff(emp_history):
    # Extract all start years and end years from the employment history list
    if len(emp_history) == 0:
        return 0
    
    end_years = []
    for entry in emp_history:
        if entry['end_year'] == None:
            end_years.append(2025)
        else:
            end_years.append(entry['end_year'])
    start_years = [entry['start_year'] for entry in emp_history]
    # end_years = [entry['end_year'] for entry in emp_history]
    
    # Find the minimum start year and maximum end year
    min_start_year = min(start_years)
    max_end_year = max(end_years)
    
    if max_end_year == None:
        max_end_year = 2025
    # Calculate and return the difference
    return max_end_year - min_start_year

# Function to classify durations
def classify_duration(duration):
    if "month" in duration:
        # Extract the number of months or range
        months = [int(s) for s in duration.split() if s.isdigit()]
        if len(months) == 1:
            month_count = months[0]
        elif len(months) == 2:
            month_count = (months[0] + months[1]) // 2  # Average for ranges
        else:
            month_count = 0  # Handle "Long-Term" etc.
        
        if month_count <= 1:
            return "Short"
        elif month_count <= 6:
            return "Medium"
        else:
            return "Long"
    
    elif "week" in duration:
        # Convert weeks to months (approx 4 weeks = 1 month)
        weeks = [int(s) for s in duration.split() if s.isdigit()]
        if len(weeks) == 1:
            month_count = weeks[0] / 4  # Convert weeks to months
        else:
            month_count = 0
        
        if month_count <= 1:
            return "Short"
        elif month_count <= 6:
            return "Medium"
        else:
            return "Long"
    else:
        return "Long"  # For cases like "Long-Term"