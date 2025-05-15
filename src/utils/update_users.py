import pandas as pd
import numpy as np
import random

# Read the existing users.csv file
users_df = pd.read_csv('data/users.csv')

# Define occupations
occupations = [
    'technician', 'other', 'writer', 'executive', 'administrator',
    'student', 'lawyer', 'educator', 'scientist', 'entertainment',
    'programmer', 'librarian', 'homemaker', 'artist', 'marketing',
    'none', 'healthcare', 'retired', 'salesman', 'doctor'
]

# Generate ages with more weight towards 20-50
def generate_age():
    # Create a weighted distribution
    weights = np.zeros(53)  # 18 to 70 inclusive
    for i in range(53):
        age = i + 18
        if 20 <= age <= 50:
            weights[i] = 0.8  # Higher weight for ages 20-50
        else:
            weights[i] = 0.2  # Lower weight for other ages
    weights = weights / weights.sum()  # Normalize weights
    
    # Generate random age based on weights
    return np.random.choice(range(18, 71), p=weights)

# Add age and occupation columns
users_df['age'] = [generate_age() for _ in range(len(users_df))]
users_df['occupation'] = [random.choice(occupations) for _ in range(len(users_df))]

# Save the updated dataframe
users_df.to_csv('data/users.csv', index=False) 