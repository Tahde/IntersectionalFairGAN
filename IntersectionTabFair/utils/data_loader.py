import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    Returns the cleaned dataframe along with sensitive attributes.
    """
    # Load and preprocess data
    df = pd.read_csv(file_path)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace(r'^\s*\?+\s*$', np.nan, regex=True, inplace=True)
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.drop(columns='fnlwgt')

    # Define bins and labels for age groups
    bins = [0, 25, 60, float('inf')]  # 0-25, 25-60, 60+
    labels = ['<25', '25–60', '>60']
    df_cleaned['age_category'] = pd.cut(df_cleaned['age'], bins=bins, labels=labels, right=False)

    # Concatenate 'Sex' and 'age_category' columns
    df_cleaned['Sex_age'] = df_cleaned['sex'].astype(str) + '_' + df_cleaned['age_category'].astype(str)
    df_cleaned = df_cleaned.drop(columns=['age', 'sex', 'age_category'])
    df_cleaned = df_cleaned.replace(' ?', np.nan).dropna()

    # Reordering columns so 'income' is placed after 'Sex_age'
    columns = list(df_cleaned.columns)
    columns.remove('income')
    columns.insert(columns.index('Sex_age') + 1, 'income')
    df_cleaned = df_cleaned[columns]

    # Define sensitive and target attributes
    S = "Sex_age"
    Y = "income"
    female_categories = df_cleaned[df_cleaned['Sex_age'].str.startswith('Female')]['Sex_age'].unique().tolist()
    S_under = female_categories
    Y_desire = ">50K"

    return df_cleaned, S, Y, S_under, Y_desire


# Function to one-hot encode the data and return relevant indices for fairness calculations
def get_ohe_data(df, fairness=True, S=None, Y=None, S_under=None, Y_desire=None):
    df_int = df.select_dtypes(['float', 'integer']).values
    continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
    scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes('object')
    df_cat_names = list(df.select_dtypes('object').columns)
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))
    final_array = np.hstack((df_int, ohe_array.toarray()))

    if fairness:
        S_start_index = len(continuous_columns_list) + sum(list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
        Y_start_index = len(continuous_columns_list) + sum(list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])

        underpriv_indices = []
        priv_indices = []
        for underpriv_category in S_under:
            underpriv_index = np.where(ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)] == underpriv_category)[0][0]
            underpriv_indices.append(S_start_index + underpriv_index)

            priv_category = underpriv_category.replace("Female", "Male")
            priv_index = np.where(ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)] == priv_category)[0][0]
            priv_indices.append(S_start_index + priv_index)

        if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0] == Y_desire:
            desire_index = 0
            undesire_index = 1
        else:
            desire_index = 1
            undesire_index = 0

        return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index
    else:
        return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array


# Function to prepare data for training, handling fairness as an option
def prepare_data(df_cleaned, batch_size, with_fairness=True, S=None, Y=None, S_under=None, Y_desire=None):
    # Get one-hot encoded data with fairness-related indices
    if with_fairness:
        ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index = get_ohe_data(
            df_cleaned, fairness=True, S=S, Y=Y, S_under=S_under, Y_desire=Y_desire)
        input_dim = final_array.shape[1]
        X_train, X_test = train_test_split(final_array, test_size=0.2, random_state=42, shuffle=True)
        data_train = X_train.copy()
        data_test = X_test.copy()

        data = torch.from_numpy(data_train).float()
        train_ds = TensorDataset(data)
        train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
        return ohe, scaler, input_dim, discrete_columns_ordereddict, continuous_columns_list, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index
    else:
        ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array = get_ohe_data(df_cleaned, fairness=False)
        input_dim = final_array.shape[1]
        X_train, X_test = train_test_split(final_array, test_size=0.2, shuffle=True)
        data_train = X_train.copy()
        data_test = X_test.copy()

        data = torch.from_numpy(data_train).float()
        train_ds = TensorDataset(data)
        train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
        return ohe, scaler, input_dim, discrete_columns_ordereddict, continuous_columns_list, train_dl, data_train, data_test


def get_original_data(df_transformed, df_orig, ohe, scaler):
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
    df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
    return pd.concat([df_int, df_cat], axis=1)
