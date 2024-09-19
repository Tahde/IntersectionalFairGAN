#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import pandas as pd
from utils.data_loader import load_and_preprocess_data, prepare_data
from utils.training import multiple_runs
from utils.metrics import set_seed

# Main function
def main():
    # Parameters
    batch_size = 256
    epochs = 200
    fair_epochs = 30
    lamda_values = np.arange(0, 2.05, 0.05)
    out_file = "results_sub_final_one_adult_new.log"
    fairness_model_path = "fairness_models_adult"
    general_model_path = "general_models_adult"
    
    # Create directories for saving models
    os.makedirs(general_model_path, exist_ok=True)
    os.makedirs(fairness_model_path, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(42)

    # Load and preprocess the data
    file_path = 'adult.csv'
    df_cleaned, S, Y, S_under, Y_desire = load_and_preprocess_data(file_path)

    # Run the training process
    multiple_runs(
        num_trainings=10,
        df=df_cleaned,
        epochs=epochs,
        batchsize=batch_size,
        fair_epochs=fair_epochs,
        lamda_values=lamda_values,
        outFile=out_file,
        fairness_model_path=fairness_model_path
    )

# Execute the main function if the script is called directly
if __name__ == "__main__":
    main()
