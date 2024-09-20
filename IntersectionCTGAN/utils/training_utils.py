#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/training_utils.py
from utils.seed_utils import set_random_seeds
from models.ctgan_modified import CTGAN
from sklearn.tree import DecisionTreeClassifier
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.base_module import BaseSynthesizer
from models.generator import Generator
from models.discriminator import Discriminator
from utils.model_utils import save_model, load_model, save_optimizer, load_optimizer
from utils.data_sampler import DataSampler
from utils.data_transformer import DataTransformer
from utils.metrics_utils import evaluate_model, calculate_demographic_parity, calculate_metrics_per_subgroup

from utils.metrics_utils import calculate_metrics_per_subgroup, SUBGROUPS
from sklearn.metrics import accuracy_score, f1_score
from utils.metrics_utils import calculate_dp_per_pair
from utils.file_utils import print2file

model_dir = 'models_ctgan_300+20'
# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)


def train_plot(df, total_epochs, batchsize, fair_epochs, lamda, discrete_columns, general_model_paths=None, save_models=None, model_paths=None, fairness_model_paths=None, seed=None):
    if seed is not None:
        set_random_seeds(seed)

    lambda_fairness = lamda
    general_epochs = total_epochs if lamda == 0 else 0  # Treat all epochs as general epochs for lambda = 0

    ctgan = CTGAN(lambda_fairness=lambda_fairness,
                  embedding_dim=128,
                  generator_dim=(256, 256),
                  discriminator_dim=(256, 256),
                  batch_size=batchsize,
                  epochs=general_epochs,
                  apply_fairness_constraint=False,  # Start without fairness constraint
                  underpriv_indices=[130, 133, 135],
                  priv_indices=[131, 132, 134],
                  Y_start_index=136,
                  desire_index=1,  # Only one index for desired outcome
                  target_attribute_index=[136, 137],
                  cuda=True)  # Use CUDA if available

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    ctgan._transformer = DataTransformer()
    ctgan._transformer.fit(train_data, discrete_columns)
    transformed_data = ctgan._transformer.transform(train_data)
    ctgan._data_sampler = DataSampler(transformed_data, ctgan._transformer.output_info_list, ctgan._log_frequency)
    input_dim = transformed_data.shape[1] - 2
# Print target columns to verify indices
    # print("Transformed Data Shape:")
    # print(transformed_data.shape)
    # print("Target Attribute Index:", ctgan.target_attribute_index)
    
    # # Extract target columns from transformed data
    # target_columns = transformed_data[:, ctgan.target_attribute_index[0]:ctgan.target_attribute_index[1] + 1]
    # print("Target Columns in Transformed Data:")
    # print(target_columns)
    # Initialize the generator and discriminator here
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

    if model_paths:
        gen_model_path, disc_model_path, optG_model_path, optD_model_path = model_paths
        if os.path.exists(gen_model_path) and os.path.exists(disc_model_path) and os.path.exists(optG_model_path) and os.path.exists(optD_model_path):
            print(f"Loading models from {gen_model_path}, {disc_model_path}, {optG_model_path}, {optD_model_path}")
            load_model(ctgan._generator, gen_model_path)
            load_model(ctgan._discriminator, disc_model_path)
            load_optimizer(optimizerG, optG_model_path, ctgan._device)
            load_optimizer(optimizerD, optD_model_path, ctgan._device)
        else:
            raise FileNotFoundError(f"Model paths {gen_model_path}, {disc_model_path}, {optG_model_path}, {optD_model_path} do not exist.")
    else:
        # Initial training for general epochs (only if lambda == 0)
        if general_epochs > 0:
            print(f"Training for general epochs: {general_epochs}")
            ctgan.fit(train_data, discrete_columns=discrete_columns, epochs=general_epochs, optimizerG=optimizerG, optimizerD=optimizerD)

            if save_models:
                gen_model_path, disc_model_path, optG_model_path, optD_model_path = save_models
                print(f"Saving models after general epochs for lambda={lamda}...")
                save_model(ctgan._generator, gen_model_path)
                save_model(ctgan._discriminator, disc_model_path)
                save_optimizer(optimizerG, optG_model_path)
                save_optimizer(optimizerD, optD_model_path)

    if fair_epochs > 0 and lamda > 0:
        # Always load the best general model for each fairness run
        if save_models:
            gen_model_path, disc_model_path, optG_model_path, optD_model_path = save_models
            print(f"Loading best general models for fairness training from {gen_model_path}, {disc_model_path}, {optG_model_path}, {optD_model_path}")
            load_model(ctgan._generator, gen_model_path)
            load_model(ctgan._discriminator, disc_model_path)
            load_optimizer(optimizerG, optG_model_path, ctgan._device)
            load_optimizer(optimizerD, optD_model_path, ctgan._device)

        # Training for fairness epochs
        print(f"Training fairness models for epochs: {fair_epochs}")
        ctgan.apply_fairness_constraint = True
        ctgan.lambda_fairness = lamda  # Ensure fairness lambda is set
        ctgan.fit(train_data, discrete_columns=discrete_columns, epochs=0, fairness_epochs=fair_epochs, optimizerG=optimizerG, optimizerD=optimizerD)

        if fairness_model_paths:
            fair_gen_model_path, fair_disc_model_path, fair_optG_model_path, fair_optD_model_path = fairness_model_paths
            print(f"Saving fairness models after epochs for lambda={lamda}...")
            save_model(ctgan._generator, fair_gen_model_path)
            save_model(ctgan._discriminator, fair_disc_model_path)
            save_optimizer(optimizerG, fair_optG_model_path)
            save_optimizer(optimizerD, fair_optD_model_path)

    return ctgan, ctgan._generator, ctgan._transformer, train_data, test_data, input_dim, optimizerG, optimizerD

def multiple_runs(num_trainings, df, total_epochs, batchsize, fair_epochs, lamda_values, outFile, model_dir, discrete_columns):
    os.makedirs(model_dir, exist_ok=True)
    fairness_model_dir = os.path.join(model_dir, 'fairness_models')
    os.makedirs(fairness_model_dir, exist_ok=True)

    # Set a fixed seed for general training phase
    fixed_seed = 42

    # Define indices pairs for demographic parity calculation
    indices_pairs = [(130, 131), (133, 132), (135, 134)]

    print("Performing general training for lambda 0.0...")
    general_model_paths_list = []
    for i in range(num_trainings):
        varied_seed = fixed_seed + i  # Slightly vary the seed for each run
        print(f"General training run {i + 1} for lambda 0.0...")

        generator_path = os.path.join(model_dir, f'generator_model_run_{i}.pth')
        discriminator_path = os.path.join(model_dir, f'discriminator_model_run_{i}.pth')
        optimizerG_path = os.path.join(model_dir, f'optimizerG_run_{i}.pth')
        optimizerD_path = os.path.join(model_dir, f'optimizerD_run_{i}.pth')

        if not os.path.exists(generator_path) or not os.path.exists(discriminator_path) or not os.path.exists(optimizerG_path) or not os.path.exists(optimizerD_path):
            ctgan, generator, transformer, train_data, test_data, input_dim, optimizerG, optimizerD = train_plot(
                df=df, total_epochs=total_epochs, batchsize=batchsize, fair_epochs=0, lamda=0, discrete_columns=discrete_columns,
                save_models=(generator_path, discriminator_path, optimizerG_path, optimizerD_path), seed=varied_seed)
        else:
            print(f"Models found for run {i + 1}, loading and evaluating...")
            ctgan, generator, transformer, train_data, test_data, input_dim, optimizerG, optimizerD = train_plot(
                df=df, total_epochs=0, batchsize=batchsize, fair_epochs=0, lamda=0, discrete_columns=discrete_columns,
                model_paths=(generator_path, discriminator_path, optimizerG_path, optimizerD_path), seed=varied_seed)

        general_model_paths_list.append((generator_path, discriminator_path, optimizerG_path, optimizerD_path))

        # Evaluate models after training or loading
        acc_original, acc_generated, f1_original, f1_generated, dp_original, dp_generated, dp_diff, original_metrics, generated_metrics, dp_pairs_original, dp_pairs_generated = evaluate_model(
            generator_path, discriminator_path, optimizerG_path, optimizerD_path, transformer, train_data, test_data, ctgan._device)
        print(f"Evaluation complete for run {i + 1}: acc_original={acc_original}, acc_generated={acc_generated}, f1_original={f1_original}, f1_generated={f1_generated}, dp_original={dp_original}, dp_generated={dp_generated}, dp_diff={dp_diff}")

        # Log initial results
        with open(outFile, 'a') as f:
            buf = '%d, %f, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f' % (
                i, 0.0, i + 1, acc_original, acc_generated, acc_original - acc_generated, f1_original, f1_generated, f1_original - f1_generated,
                dp_original, dp_generated, dp_diff)

            for subgroup in SUBGROUPS['Sex_age']:
                buf += f", {original_metrics[subgroup]['accuracy']}, {original_metrics[subgroup]['f1']}, {generated_metrics[subgroup]['accuracy']}, {generated_metrics[subgroup]['f1']}"

            for dp_orig, dp_gen in zip(dp_pairs_original, dp_pairs_generated):
                buf += f", {dp_orig}, {dp_gen}"

            print(f"Logging initial results for run {i + 1}")
            f.write(buf + '\n')

    # Evaluate models to find the best one
    best_model_idx = None
    best_score = float('inf')
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    transformer = DataTransformer()
    transformer.fit(train_data, discrete_columns)

    for i in range(num_trainings):
        acc_original, acc_generated, f1_original, f1_generated, dp_original, dp_generated, dp_diff, original_metrics, generated_metrics, dp_pairs_original, dp_pairs_generated = evaluate_model(
            general_model_paths_list[i][0], general_model_paths_list[i][1], general_model_paths_list[i][2], general_model_paths_list[i][3], transformer, train_data, test_data, ctgan._device)
        acc_diff = abs(acc_original - acc_generated)
        f1_diff = abs(f1_original - f1_generated)
        score = acc_diff + f1_diff + dp_diff
        print(f"Model {i}: acc_diff={acc_diff}, f1_diff={f1_diff}, dp_diff={dp_diff}, score={score}")
        print(f"Evaluation for run {i + 1}: acc_original={acc_original}, acc_generated={acc_generated}, f1_original={f1_original}, f1_generated={f1_generated}, dp_original={dp_original}, dp_generated={dp_generated}, dp_diff={dp_diff}")
        if score < best_score:
            best_model_idx = i
            best_score = score

    best_model_paths = general_model_paths_list[best_model_idx]
    print(f"Selected model from run {best_model_idx} for fairness training.")

    # Initialize the results file
    with open(outFile, 'a') as f:
        #first_line = f"num_trainings: {num_trainings}, total_epochs: {total_epochs}, batchsize: {batchsize}, fair_epochs: {fair_epochs}, lamda: {lamda}"
        #first_line = "num_trainings: %d, total_epochs: %d, batchsize: %d, fair_epochs: %d, lamda: %f" % (
            #num_trainings, total_epochs, batchsize, fair_epochs, lamda_values[0])
        #print(first_line)
        #f.write(first_line + '\n')

        second_line = "train_idx, lambda, iteration, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1, demographic_parity_original, demographic_parity_generated, demographic_parity_diff"
        for subgroup in SUBGROUPS['Sex_age']:
            second_line += f", acc_orig_{subgroup}, f1_orig_{subgroup}, acc_gen_{subgroup}, f1_gen_{subgroup}"
        for pair_idx in range(len(indices_pairs)):
            second_line += f", dp_orig_pair_{pair_idx}, dp_gen_pair_{pair_idx}"
        print(second_line)
        f.write(second_line + '\n')

    for lamda in lamda_values:
        if lamda == 0:
            continue  # Skip the fairness training for lambda 0
        print(f"Running with lambda: {lamda}")
        valid_runs = 0
        while valid_runs < num_trainings:
            print(f"Training iteration {valid_runs + 1} for lambda {lamda}")

            # Use unique filenames for each lambda value and run
            generator_fairness_path = os.path.join(fairness_model_dir, f'generator_lambda_{lamda:.2f}_run_{valid_runs}.pth')
            discriminator_fairness_path = os.path.join(fairness_model_dir, f'discriminator_lambda_{lamda:.2f}_run_{valid_runs}.pth')
            optimizerG_fairness_path = os.path.join(fairness_model_dir, f'optimizerG_lambda_{lamda:.2f}_run_{valid_runs}.pth')
            optimizerD_fairness_path = os.path.join(fairness_model_dir, f'optimizerD_lambda_{lamda:.2f}_run_{valid_runs}.pth')

            # Always load the best general model for each fairness run
            ctgan, generator, transformer, train_data, test_data, input_dim, optimizerG, optimizerD = train_plot(
                df=df, total_epochs=0, batchsize=batchsize, fair_epochs=fair_epochs, lamda=lamda, discrete_columns=discrete_columns,
                model_paths=best_model_paths)

            data_train_x = transformer.transform(train_data)[:, :-2]
            data_train_y = transformer.transform(train_data)[:, -2:]
            data_test_x = transformer.transform(test_data)[:, :-2]
            data_test_y = transformer.transform(test_data)[:, -2:]

            fakez = torch.normal(mean=torch.zeros(ctgan._batch_size, ctgan._embedding_dim, device=ctgan._device),
                                 std=torch.ones(ctgan._batch_size, ctgan._embedding_dim, device=ctgan._device))
            condvec = ctgan._data_sampler.sample_condvec(ctgan._batch_size)
            if condvec is not None:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(ctgan._device)
                fakez = torch.cat([fakez, c1], dim=1)
            fake = ctgan._generator(fakez)
            fakeact = ctgan._apply_activate(fake)
            synthetic_data = fakeact.detach().cpu().numpy()

            generated_x = synthetic_data[:, :-2]
            generated_y = synthetic_data[:, -2:]

            print("Training classifier on original data")
            clf = DecisionTreeClassifier()
            clf.fit(data_train_x, np.argmax(data_train_y, axis=1))
            accuracy_original = accuracy_score(clf.predict(data_test_x), np.argmax(data_test_y, axis=1))
            f1_original = f1_score(clf.predict(data_test_x), np.argmax(data_test_y, axis=1))

            print("Training classifier on generated data")
            clf_generated = DecisionTreeClassifier()
            clf_generated.fit(generated_x, np.argmax(generated_y, axis=1))
            accuracy_generated = accuracy_score(clf_generated.predict(data_test_x), np.argmax(data_test_y, axis=1))
            f1_generated = f1_score(clf_generated.predict(data_test_x), np.argmax(data_test_y, axis=1))

            difference_accuracy = accuracy_original - accuracy_generated
            difference_f1 = f1_original - f1_generated

            demographic_parity_original = calculate_demographic_parity(data_train_x, np.argmax(data_train_y, axis=1), indices_pairs)
            demographic_parity_generated = calculate_demographic_parity(generated_x, np.argmax(generated_y, axis=1), indices_pairs)
            demographic_parity_diff = demographic_parity_original - demographic_parity_generated

            dp_pairs_original = calculate_dp_per_pair(data_train_x, np.argmax(data_train_y, axis=1), indices_pairs)
            dp_pairs_generated = calculate_dp_per_pair(generated_x, np.argmax(generated_y, axis=1), indices_pairs)

            #if difference_f1 > 0.25 or demographic_parity_diff < -0.10:
                #print(f"Skipping run {valid_runs + 1} due to high difference in F1 score ({difference_f1}) or high demographic parity difference #({demographic_parity_diff})")
                #continue

            original_metrics = calculate_metrics_per_subgroup(clf, data_test_x, np.argmax(data_test_y, axis=1), SUBGROUPS['Sex_age'])
            generated_metrics = calculate_metrics_per_subgroup(clf_generated, data_test_x, np.argmax(data_test_y, axis=1), SUBGROUPS['Sex_age'])

            buf = '%d, %f, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f' % (
                valid_runs, lamda, valid_runs + 1, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1,
                demographic_parity_original, demographic_parity_generated, demographic_parity_diff)

            for subgroup in SUBGROUPS['Sex_age']:
                buf += f", {original_metrics[subgroup]['accuracy']}, {original_metrics[subgroup]['f1']}, {generated_metrics[subgroup]['accuracy']}, {generated_metrics[subgroup]['f1']}"

            for dp_orig, dp_gen in zip(dp_pairs_original, dp_pairs_generated):
                buf += f", {dp_orig}, {dp_gen}"

            print(f"Logging results for iteration {valid_runs + 1} with lambda {lamda}")
            print(buf)
            print2file(buf, outFile)

            # Save the models after each fairness training iteration
            save_model(ctgan._generator, generator_fairness_path)
            save_model(ctgan._discriminator, discriminator_fairness_path)
            save_optimizer(optimizerG, optimizerG_fairness_path)
            save_optimizer(optimizerD, optimizerD_fairness_path)

            valid_runs += 1

    print("Training and evaluation complete. Results have been logged to the file.")
