import os
import torch
import pandas as pd
from torch import optim
from utils.model_utils import load_model, load_optimizer
from utils.data_utils import load_and_preprocess_data
from utils.data_transformer import DataTransformer
from utils.data_sampler import DataSampler
from models.ctgan_modified import CTGAN
from models.generator import Generator
from models.discriminator import Discriminator
from sklearn.model_selection import train_test_split

def main():
    # Set the paths to your models inside the Best_models directory
    best_models_dir = "Best_models"

    best_generator_path = os.path.join(best_models_dir, "generator_model_run_2.pth")
    best_discriminator_path = os.path.join(best_models_dir, "discriminator_model_run_2.pth")
    best_optimizerG_path = os.path.join(best_models_dir, "optimizerG_run_2.pth")
    best_optimizerD_path = os.path.join(best_models_dir, "optimizerD_run_2.pth")

    fairness_generator_path = os.path.join(best_models_dir, "generator_lambda_0.30_run_6.pth")
    fairness_discriminator_path = os.path.join(best_models_dir, "discriminator_lambda_0.30_run_6.pth")
    fairness_optimizerG_path = os.path.join(best_models_dir, "optimizerG_lambda_0.30_run_6.pth")
    fairness_optimizerD_path = os.path.join(best_models_dir, "optimizerD_lambda_0.30_run_6.pth")

    # Load and preprocess the data using your data_utils module
    file_path = 'adult.csv'
    df_cleaned = load_and_preprocess_data(file_path)

    discrete_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'native-country', 'Sex_age', 'income']
    
    # Train-test split
    train_data, _ = train_test_split(df_cleaned, test_size=0.2, random_state=42)

    # Initialize data transformer and sampler
    transformer = DataTransformer()
    transformer.fit(train_data, discrete_columns)
    transformed_data = transformer.transform(train_data)
    data_sampler = DataSampler(transformed_data, transformer.output_info_list, log_frequency=True)

    # Initialize the CTGAN model
    ctgan = CTGAN(lambda_fairness=0.3,
                  embedding_dim=128,
                  generator_dim=(256, 256),
                  discriminator_dim=(256, 256),
                  batch_size=500,
                  epochs=0,
                  apply_fairness_constraint=False,  # Start without fairness constraint
                  underpriv_indices=[130, 133, 135],
                  priv_indices=[131, 132, 134],
                  Y_start_index=136,
                  desire_index=1,  # Only one index for desired outcome
                  target_attribute_index=[136, 137],
                  cuda=True)  # Use CUDA if available

    ctgan._transformer = transformer
    ctgan._data_sampler = data_sampler
    ctgan._generator = Generator(
        ctgan._embedding_dim + ctgan._data_sampler.dim_cond_vec(),
        ctgan._generator_dim,
        ctgan._transformer.output_dimensions
    ).to(ctgan._device)

    ctgan._discriminator = Discriminator(
        ctgan._transformer.output_dimensions + ctgan._data_sampler.dim_cond_vec(),
        ctgan._discriminator_dim,
        pac=ctgan.pac
    ).to(ctgan._device)

    optimizerG = torch.optim.Adam(ctgan._generator.parameters(), lr=ctgan._generator_lr, betas=(0.5, 0.9), weight_decay=ctgan._generator_decay)
    optimizerD = torch.optim.Adam(ctgan._discriminator.parameters(), lr=ctgan._discriminator_lr, betas=(0.5, 0.9), weight_decay=ctgan._discriminator_decay)

    # Load the best general model
    load_model(ctgan._generator, best_generator_path)
    load_model(ctgan._discriminator, best_discriminator_path)
    load_optimizer(optimizerG, best_optimizerG_path, ctgan._device)
    load_optimizer(optimizerD, best_optimizerD_path, ctgan._device)

    # Load the fairness model
    load_model(ctgan._generator, fairness_generator_path)
    load_model(ctgan._discriminator, fairness_discriminator_path)
    load_optimizer(optimizerG, fairness_optimizerG_path, ctgan._device)
    load_optimizer(optimizerD, fairness_optimizerD_path, ctgan._device)

    # Generate synthetic data
    synthetic_data = ctgan.sample(len(train_data))  # Generate synthetic data matching the size of the training data

    output_dir = 'generated_csv_files'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ctgan-adult-final-6.csv')
    
    pd.DataFrame(synthetic_data, columns=df_cleaned.columns).to_csv(output_file, index=False)
    print(f"Synthetic data saved to '{output_file}'.")

if __name__ == "__main__":
    main()
