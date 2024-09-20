#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from utils.training_utils import multiple_runs  # Import from your training utils
from utils.data_utils import load_and_preprocess_data  # Import the data preprocessing function
from utils.seed_utils import set_random_seeds  # Import seed setting utility for reproducibility
from utils.metrics_utils import calculate_metrics_per_subgroup, SUBGROUPS

# Load and preprocess the data
file_path = 'adult.csv'
df_cleaned = load_and_preprocess_data(file_path)

# Define training hyperparameters
num_trainings = 10
total_epochs = 300
batchsize = 500
fairness_epochs = 20
lamda_values = np.arange(0, 2.1, 0.1)  # Range of lambda values for fairness
outFile = 'models_ctgan_300+20.txt'  # Output file for logging results
model_dir = 'models_ctgan_300+20'  # Directory for saving models

# Specify the columns that will be treated as discrete
discrete_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'native-country', 'Sex_age', 'income']

# Run the training and evaluation pipeline with multiple lambda values
multiple_runs(num_trainings, df_cleaned, total_epochs, batchsize, fairness_epochs, 
              lamda_values, outFile, model_dir, discrete_columns)

