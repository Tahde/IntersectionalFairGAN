#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from utils.data_transformer import DataTransformer
from sklearn.model_selection import train_test_split
from utils.data_utils import load_and_preprocess_data

file_path = 'adult.csv'
df_cleaned = load_and_preprocess_data(file_path)
def find_transformed_feature_indices(column_transform_info_list, target_column):
    """
    Find the start and end indices for transformed features of a specific column.
    """
    start_index = 0
    num_unique_values = 6  # unique values for Sex_age (as an example)

    for info in column_transform_info_list:
        if info.column_name == target_column:
            num_unique_values = info.output_dimensions  # Get number of unique categories for the target column
            break
        else:
            start_index += info.output_dimensions
    
    end_index = start_index + num_unique_values
    return start_index, end_index

def identify_sensitive_indices(train_data, transformer):
    """
    Identify the transformed feature indices for sensitive attributes and map categories.
    """
    unique_sex_age_categories = train_data['Sex_age'].unique()
    print("Unique 'Sex_age' categories:", unique_sex_age_categories)

    # Find the transformed feature indices for 'Sex_age'
    start_pos, end_pos = find_transformed_feature_indices(transformer._column_transform_info_list, 'Sex_age')
    print(f"Start index for 'Sex_Income' transformed features: {start_pos}")
    print(f"End index for 'Sex_Income' transformed features: {end_pos}")
    
    # Map 'Sex_age' categories to their indices
    category_to_index = {category: idx for idx, category in enumerate(unique_sex_age_categories)}
    
    # Define underprev and prev categories
    underprev_categories = ['Female_25–60', 'Female_<25', 'Female_>60']  # underprivileged categories
    prev_categories = ['Male_25–60', 'Male_<25', 'Male_>60']  # privileged categories
    
    underprev_indices = [category_to_index[cat] for cat in underprev_categories if cat in category_to_index]
    prev_indices = [category_to_index[cat] for cat in prev_categories if cat in category_to_index]
    
    print("Category to index mapping:", category_to_index)
    print("Underprev indices:", underprev_indices)
    print("Prev indices:", prev_indices)
    
    return underprev_indices, prev_indices

def map_to_transformed_positions(underprev_indices, prev_indices, start_index):
    """
    Map underprivileged and privileged indices to positions within transformed features.
    """
    underprev_positions = [start_index + idx for idx in underprev_indices]
    prev_positions = [start_index + idx for idx in prev_indices]
    
    print("Underprev positions within transformed features:", underprev_positions)
    print("Prev positions within transformed features:", prev_positions)
    
    return underprev_positions, prev_positions

# Example usage
if __name__ == "__main__":
    file_path = 'adult.csv'
    df_cleaned = load_and_preprocess_data(file_path)

    discrete_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'Sex_age', 'income']
    
    # Split the data
    train_data, _ = train_test_split(df_cleaned, test_size=0.2, random_state=42)
    
    # Initialize the transformer
    transformer = DataTransformer()
    transformer.fit(train_data, discrete_columns)
    
    # Identify sensitive attribute indices
    underprev_indices, prev_indices = identify_sensitive_indices(train_data, transformer)
    
    # Map them to transformed feature positions
    start_index = 130  # Start index for Sex_age
    map_to_transformed_positions(underprev_indices, prev_indices, start_index)

