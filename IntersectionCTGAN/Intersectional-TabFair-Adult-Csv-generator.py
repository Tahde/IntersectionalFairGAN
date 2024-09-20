#!/usr/bin/env python
# coding: utf-8

import torch
import os
from utils.data_loader import load_and_preprocess_data, prepare_data, get_original_data
from models.generator import Generator
from models.critic import Critic
from models.loss import FairLossFunc
from utils.data_loader import get_ohe_data
from torch.utils.data import TensorDataset, DataLoader


def generate_data(file_path, batch_size=256, lamda=1.75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the data
    df_cleaned, S, Y, S_under, Y_desire = load_and_preprocess_data(file_path)
    
    # Prepare data for fairness
    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index = prepare_data(
        df_cleaned, batch_size, with_fairness=True, S=S, Y=Y, S_under=S_under, Y_desire=Y_desire)

    # Initialize models
    generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    fair_loss_func = FairLossFunc(S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index).to(device)

    # Optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Paths for saving/loading models from 'best_models' folder
    best_model_folder = './Best_models'
    general_generator_path = os.path.join(best_model_folder, 'generator_pre_fairness_new_run_9.pt')
    general_critic_path = os.path.join(best_model_folder, 'critic_pre_fairness_new_run_9.pt')
    fairness_generator_path = os.path.join(best_model_folder, f'generator_run_6_lambda_{lamda}.pt')
    fairness_critic_path = os.path.join(best_model_folder, f'critic_run_6_lambda_{lamda}.pt')

    # Load pre-trained models if available
    if os.path.exists(general_generator_path) and os.path.exists(general_critic_path):
        generator.load_state_dict(torch.load(general_generator_path))
        critic.load_state_dict(torch.load(general_critic_path))
        generator.to(device)
        critic.to(device)
        print("Pre-trained general models loaded from 'best_models'.")

    if os.path.exists(fairness_generator_path) and os.path.exists(fairness_critic_path):
        generator.load_state_dict(torch.load(fairness_generator_path))
        critic.load_state_dict(torch.load(fairness_critic_path))
        generator.to(device)
        critic.to(device)
        print("Fairness models loaded from 'best_models'.")
    else:
        raise FileNotFoundError("No fairness models found in 'best_models'.")

    # Generate data using the trained generator
    num_samples = int(0.8 * len(df_cleaned))
    fake_data = generator(torch.randn(size=(num_samples, input_dim), device=device)).cpu().detach().numpy()

    # Convert generated data back to original format
    original_data = get_original_data(fake_data, df_cleaned, ohe, scaler)

    # Ensure the folder exists for saving the generated CSV files
    output_folder = 'generated_csv_files'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the generated data to the folder
    output_file = os.path.join(output_folder, 'adult_generated_data.csv')
    original_data.to_csv(output_file, index=False)
    print(f"Generated data saved to {output_file}")

if __name__ == "__main__":
    file_path = 'adult.csv'  # Provide the path to your dataset here
    generate_data(file_path)
