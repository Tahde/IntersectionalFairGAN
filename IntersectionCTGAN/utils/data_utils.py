#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/data_utils.py
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Preprocess the data (clean, strip whitespace, handle missing values)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace(r'^\s*\?+\s*$', np.nan, regex=True, inplace=True)
    df_cleaned = df.dropna().drop(columns=['fnlwgt'])
    
    # Binning age into categories and creating combined columns
    bins = [0, 25, 60, float('inf')]
    labels = ['<25', '25–60', '>60']
    df_cleaned['age_category'] = pd.cut(df_cleaned['age'], bins=bins, labels=labels, right=False)
    df_cleaned['Sex_age'] = df_cleaned['sex'].astype(str) + '_' + df_cleaned['age_category'].astype(str)
    df_cleaned = df_cleaned.drop(columns=['age', 'sex', 'age_category'])
    
    # Final clean-up
    df_cleaned = df_cleaned.replace(' ?', np.nan).dropna()

    # Reorder the columns to have 'Sex_age' followed by 'income'
    columns = list(df_cleaned.columns)
    columns.remove('income')
    columns.insert(columns.index('Sex_age') + 1, 'income')
    df_cleaned = df_cleaned[columns]
    
    return df_cleaned


