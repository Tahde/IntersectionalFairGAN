#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import torch
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from models.generator import Generator
from models.critic import Critic
from utils.data_loader import prepare_data
from utils.metrics import calculate_demographic_parity, calculate_subgroup_metrics
from utils.metrics import set_seed
from utils.helpers import save_model, load_model, print2file, log_subgroup_metrics,  log_dp_metrics
from utils.data_loader import load_and_preprocess_data

import warnings
from models.loss import FairLossFunc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
import torch

# Define the functions in this module
def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss

# Set initial parameters
batch_size = 256
num_epochs = 200
num_fair_epochs = 30
lambda_val = 0.2

# Load the data
file_path = 'adult.csv'  # Path to the dataset
df_cleaned, S, Y, S_under, Y_desire = load_and_preprocess_data(file_path)

# Use df_cleaned, S, Y, S_under, Y_desire with prepare_data function
ohe, scaler, input_dim, discrete_columns_ordereddict, continuous_columns_list, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index = prepare_data(
    df_cleaned, batch_size, with_fairness=True, S=S, Y=Y, S_under=S_under, Y_desire=Y_desire
)
def train(df, epochs=200, batch_size=64, fair_epochs=30, lamda=0.5, save_general_models=False, save_fairness_models=False, fairness_model_path=None, run_id=None, best_run=None):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #general_generator_path = f'generator_pre_fairness_new_run_{run_id}.pt'
    #general_critic_path = f'critic_pre_fairness_new_run_{run_id}.pt'
    general_generator_path = f'./general_models_adult/generator_pre_fairness_new_run_{run_id}.pt'
    general_critic_path = f'./general_models_adult/critic_pre_fairness_new_run_{run_id}.pt'
    fairness_generator_path = f'{fairness_model_path}/run_{run_id}_lambda_{lamda}/generator_run_{run_id}_lambda_{lamda}.pt'
    fairness_critic_path = f'{fairness_model_path}/run_{run_id}_lambda_{lamda}/critic_run_{run_id}_lambda_{lamda}.pt'
    best_general_generator_path = f'./general_models_adult/generator_pre_fairness_new_run_{best_run}.pt'
    best_general_critic_path = f'./general_models_adult/critic_pre_fairness_new_run_{best_run}.pt'
    models_loaded = False
    fairness_models_loaded = False

    if lamda == 0:
        fair_epochs = 0

    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index = prepare_data(
        df_cleaned, batch_size, with_fairness=True, S=S, Y=Y, S_under=S_under, Y_desire=Y_desire)
    
    generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    fair_loss_func = FairLossFunc(S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index).to(device) 
    
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load pre-trained general models if they exist and lambda == 0
    if lamda == 0 and os.path.exists(general_generator_path) and os.path.exists(general_critic_path):
        generator.load_state_dict(torch.load(general_generator_path))
        critic.load_state_dict(torch.load(general_critic_path))
        generator.to(device)
        critic.to(device)
        models_loaded = True
        print(f"Pre-trained general models for run {run_id} loaded.")
    elif lamda > 0 and os.path.exists(best_general_generator_path) and os.path.exists(best_general_critic_path):
        # Load best pre-trained general models for fairness training if lambda > 0
        generator.load_state_dict(torch.load(best_general_generator_path))
        critic.load_state_dict(torch.load(best_general_critic_path))
        generator.to(device)
        critic.to(device)
        models_loaded = True
        print(f"Loaded best general models for fairness training with lambda {lamda}.")
    else:
        print(f"No pre-trained models found, starting training from scratch.")

    # Load pre-trained fairness models if they exist and lambda > 0
    if lamda > 0 and os.path.exists(fairness_generator_path) and os.path.exists(fairness_critic_path):
        generator.load_state_dict(torch.load(fairness_generator_path))
        critic.load_state_dict(torch.load(fairness_critic_path))
        generator.to(device)
        critic.to(device)
        fairness_models_loaded = True
        print(f"Pre-trained fairness models for lambda {lamda} and run {run_id} found, skipping training.")

    if models_loaded and lamda == 0:
        return generator, critic, ohe, scaler, data_train, data_test, input_dim  # Skip training if general models are loaded for lambda 0

    if fairness_models_loaded:
        return generator, critic, ohe, scaler, data_train, data_test, input_dim  # Skip training if fairness models are loaded

    cur_step = 0
    start_epoch = epochs - fair_epochs if models_loaded and lamda > 0 else 0
    for epoch in range(start_epoch, epochs):
        if lamda > 0 and epoch >= epochs - fair_epochs:
            print(f"Fairness Training Run {epoch - (epochs - fair_epochs) + 1}")

        for data in train_dl:
            crit_repeat = 4
            data[0] = data[0].to(device)
            
            for k in range(crit_repeat):
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake = generator(fake_noise)
                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])
                epsilon = torch.rand(batch_size, 1, device=device, requires_grad=True)
                gradient = get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()

            fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
            fake_2 = generator(fake_noise_2)
            crit_fake_pred_2 = critic(fake_2)
            
            if lamda > 0 and epoch >= epochs - fair_epochs:
                current_lambda = lamda 
                gen_optimizer_fair.zero_grad()
                combined_loss = fair_loss_func(fake_2, crit_fake_pred_2, current_lambda)
                combined_loss.backward()
                gen_optimizer_fair.step()
                
            else:
                current_lambda = 0 
                gen_optimizer.zero_grad()
                gen_loss = get_gen_loss(crit_fake_pred_2)
                gen_loss.backward()
                gen_optimizer.step()
            
            cur_step += 1 

    # Save general models if lambda is 0
    if save_general_models and lamda == 0 and not models_loaded:
        save_model(generator, general_generator_path)
        save_model(critic, general_critic_path)

    # Save fairness models if lambda is not 0
    if save_fairness_models and lamda > 0:
        run_model_path = os.path.join(fairness_model_path, f'run_{run_id}_lambda_{lamda}')
        os.makedirs(run_model_path, exist_ok=True)
        save_model(generator, fairness_generator_path)
        save_model(critic, fairness_critic_path)

    return generator, critic, ohe, scaler, data_train, data_test, input_dim

def train_plot(df, epochs, batchsize, fair_epochs, lamda, save_models_once, fairness_model_path, run_id, best_run):
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train(df, epochs, batchsize, fair_epochs, lamda, save_models_once, save_models_once, fairness_model_path, run_id, best_run)
    return generator, critic, ohe, scaler, data_train, data_test, input_dim

def evaluate_models(num_trainings, df, batchsize, fair_epochs, general_model_path, ohe, scaler, input_dim, discrete_columns_ordereddict, continuous_columns_list, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index):
    evaluations = []

    for run_id in range(num_trainings):
        #general_generator_path = f'generator_pre_fairness_new_run_{run_id}.pt'
        #general_critic_path = f'critic_pre_fairness_new_run_{run_id}.pt'
        general_generator_path = f'./general_models_adult/generator_pre_fairness_new_run_{run_id}.pt'
        general_critic_path = f'./general_models_adult/critic_pre_fairness_new_run_{run_id}.pt'   

        if not (os.path.exists(general_generator_path) and os.path.exists(general_critic_path)):
            continue  # Skip if the general model doesn't exist

        

        generator = Generator(input_dim, continuous_columns_list, discrete_columns_ordereddict).to(device)
        critic = Critic(input_dim).to(device)
        load_model(generator, general_generator_path)
        load_model(critic, general_critic_path)
        generator.eval()
        critic.eval()
        df_generated = generator(torch.randn(size=(32561, input_dim), device=device)).cpu().detach().numpy()
        df_generated_x = df_generated[:, :-2]
        df_generated_y = np.argmax(df_generated[:, -2:], axis=1)

        data_train_x = data_train[:, :-2]
        data_train_y = np.argmax(data_train[:, -2:], axis=1)

        data_test_x = data_test[:, :-2]
        data_test_y = np.argmax(data_test[:, -2:], axis=1)

        clf = DecisionTreeClassifier()
        clf.fit(data_train_x, data_train_y)
        accuracy_original = accuracy_score(clf.predict(data_test_x), data_test_y)
        f1_original = f1_score(clf.predict(data_test_x), data_test_y)
        clf_generated = DecisionTreeClassifier()
        clf_generated.fit(df_generated_x, df_generated_y)
        accuracy_generated = accuracy_score(clf_generated.predict(data_test_x), data_test_y)
        f1_generated = f1_score(clf_generated.predict(data_test_x), data_test_y)

        difference_accuracy = abs(accuracy_original - accuracy_generated)
        difference_f1 = abs(f1_original - f1_generated)

        indices_pairs = list(zip(underpriv_indices, priv_indices))
        dp_original = calculate_demographic_parity(data_train_x, data_train_y, indices_pairs)
        dp_generated = calculate_demographic_parity(df_generated_x, df_generated_y, indices_pairs)

        dp_diff = abs(np.mean([d[2] for d in dp_original]) - np.mean([d[2] for d in dp_generated]))

        total_difference = difference_accuracy + difference_f1 + dp_diff
        print(f"Run {run_id}: accuracy_diff={difference_accuracy}, f1_diff={difference_f1}, dp_diff={dp_diff}, total_diff={total_difference}")
        evaluations.append((run_id, total_difference))
    print(f"Evaluations: {evaluations}")
    # Select the best run based on the lowest sum of differences
    best_run = min(evaluations, key=lambda x: x[1])[0]
    print(f"Best run based on lowest total difference: {best_run}")
    return best_run



set_seed(42)
def multiple_runs(num_trainings, df, epochs, batchsize, fair_epochs, lamda_values, outFile, fairness_model_path):
    first_fairness_run = True  # Flag to handle the first fairness run
    best_run = None

    for lamda in lamda_values:
        save_general_models = (lamda == 0)  # Save general models only for lambda=0
        save_fairness_models = first_fairness_run and (lamda > 0)  # Save fairness models only for the first lambda > 0

        if lamda == 0:
            for i in range(num_trainings):
                # Set a different random seed for each run
                set_seed(42 + i)

                if i == 0:
                    first_line = "Lambda: %f, num_trainings: %d, num_epochs: %d, batchsize: %d, fair_epochs:%d" % (
                        lamda, num_trainings, epochs, batchsize, fair_epochs)
                    print(first_line)
                    print2file(first_line, outFile)
                    second_line = "train_idx, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1, dp_original, dp_generated"
                    print(second_line)
                    print2file(second_line, outFile)

                generator, critic, ohe, scaler, data_train, data_test, input_dim = train(df, epochs, batchsize, fair_epochs, lamda, save_general_models, save_fairness_models, fairness_model_path, run_id=i)
                save_general_models = False  # Ensure models are only saved once during general training

                # Original data metrics
                data_train_x = data_train[:, :-2]
                data_train_y = np.argmax(data_train[:, -2:], axis=1)

                data_test_x = data_test[:, :-2]
                data_test_y = np.argmax(data_test[:, -2:], axis=1)

                # Generated data metrics
                df_generated = generator(torch.randn(size=(32561, input_dim), device=device)).cpu().detach().numpy()
                df_generated_x = df_generated[:, :-2]
                df_generated_y = np.argmax(df_generated[:, -2:], axis=1)

                clf = DecisionTreeClassifier()  # Re-initialize the classifier for original data
                clf.fit(data_train_x, data_train_y)  # Train on original data
                data_pred_original = clf.predict(data_test_x)  # Predictions on original test data
                accuracy_original = accuracy_score(data_pred_original, data_test_y)
                f1_original = f1_score(data_pred_original, data_test_y, zero_division=0)

                # Train on generated data
                clf_generated = DecisionTreeClassifier()  # Re-initialize the classifier for generated data
                clf_generated .fit(df_generated_x, df_generated_y)  # Train on generated data
                data_pred_gen =  clf_generated.predict(data_test_x)  # Predictions on the same test data
                accuracy_generated = accuracy_score(data_pred_gen, data_test_y)
                f1_generated = f1_score(data_pred_gen, data_test_y, zero_division=0)

                difference_accuracy = accuracy_original - accuracy_generated
                difference_f1 = f1_original - f1_generated

                indices_pairs = list(zip(underpriv_indices, priv_indices))
                dp_original = calculate_demographic_parity(data_train_x, data_train_y, indices_pairs)
                dp_generated = calculate_demographic_parity(df_generated_x, df_generated_y, indices_pairs)

                buf = '%d, %f, %f, %f, %f, %f, %f, %f, %f' % (
                    i, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1, np.mean([d[2] for d in dp_original]), np.mean([d[2] for d in dp_generated]))
                print(buf)
                print2file(buf, outFile)

                # Subgroup metrics for original data
                subgroups = ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)]
                data_pred_original = clf.predict(data_test_x)
                data_pred_gen = clf_generated.predict(data_test_x)
                subgroup_metrics_original = calculate_subgroup_metrics(data_test_x, data_test_y, data_pred_original, subgroups, S_start_index)
                log_subgroup_metrics(subgroup_metrics_original, "Original", outFile)

                # Log DP for each pair
                log_dp_metrics(dp_original, "Original", subgroups, S_start_index, outFile)
                log_dp_metrics(dp_generated, "Generated", subgroups, S_start_index, outFile)

                # Subgroup metrics for generated data
                subgroup_metrics_generated = calculate_subgroup_metrics(data_test_x, data_test_y, data_pred_gen, subgroups, S_start_index)
                log_subgroup_metrics(subgroup_metrics_generated, "Generated", outFile)

                # Save the models for each run and lambda
                if lamda == 0:
                    save_model(generator, f'generator_pre_fairness_new_run_{i}.pt')
                    save_model(critic, f'critic_pre_fairness_new_run_{i}.pt')

            best_run = evaluate_models(num_trainings, df, batchsize, fair_epochs, None, ohe, scaler, input_dim, discrete_columns_ordereddict, continuous_columns_list, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index)
            print(f"Best run: {best_run}")
        else:
            for i in range(num_trainings):
                fairness_generator_path = f'{fairness_model_path}/run_{i}_lambda_{lamda}/generator_run_{i}_lambda_{lamda}.pt'
                fairness_critic_path = f'{fairness_model_path}/run_{i}_lambda_{lamda}/critic_run_{i}_lambda_{lamda}.pt'
                
                if os.path.exists(fairness_generator_path) and os.path.exists(fairness_critic_path):
                    print(f"Pre-trained fairness models for lambda {lamda} and run {i} found, skipping training.")
                    #generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
                    generator = Generator(input_dim, continuous_columns_list, discrete_columns_ordereddict).to(device)
                    critic = Critic(input_dim).to(device)
                    load_model(generator, fairness_generator_path)
                    load_model(critic, fairness_critic_path)
                else:
                    if i == 0:
                        first_line = "Lambda: %f, num_trainings: %d, num_epochs: %d, batchsize: %d, fair_epochs:%d" % (
                            lamda, num_trainings, epochs, batchsize, fair_epochs)
                        print(first_line)
                        print2file(first_line, outFile)
                        second_line = "train_idx, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1, dp_original, dp_generated"
                        print(second_line)
                        print2file(second_line, outFile)

                    generator, critic, ohe, scaler, data_train, data_test, input_dim = train(df, epochs, batchsize, fair_epochs, lamda, save_general_models, save_fairness_models, fairness_model_path, run_id=i, best_run=best_run)
                    save_general_models = False  # Ensure models are only saved once during general training

                    if first_fairness_run and lamda > 0:
                        first_fairness_run = False  # Set the flag to False after the first fairness run

                # Original data metrics
                data_train_x = data_train[:, :-2]
                data_train_y = np.argmax(data_train[:, -2:], axis=1)

                data_test_x = data_test[:, :-2]
                data_test_y = np.argmax(data_test[:, -2:], axis=1)

                # Generated data metrics
                df_generated = generator(torch.randn(size=(32561, input_dim), device=device)).cpu().detach().numpy()
                df_generated_x = df_generated[:, :-2]
                df_generated_y = np.argmax(df_generated[:, -2:], axis=1)
                  
                clf = DecisionTreeClassifier()
                clf.fit(data_train_x, data_train_y)
                accuracy_original = accuracy_score(clf.predict(data_test_x), data_test_y)
                f1_original = f1_score(clf.predict(data_test_x), data_test_y, zero_division=0)
                clf_generated = DecisionTreeClassifier()
                clf_generated.fit(df_generated_x, df_generated_y)
                accuracy_generated = accuracy_score(clf_generated.predict(data_test_x), data_test_y)
                f1_generated = f1_score(clf_generated.predict(data_test_x), data_test_y, zero_division=0)

                difference_accuracy = accuracy_original - accuracy_generated
                difference_f1 = f1_original - f1_generated

                indices_pairs = list(zip(underpriv_indices, priv_indices))
                dp_original = calculate_demographic_parity(data_train_x, data_train_y, indices_pairs)
                dp_generated = calculate_demographic_parity(df_generated_x, df_generated_y, indices_pairs)

                buf = '%d, %f, %f, %f, %f, %f, %f, %f, %f' % (
                    i, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1, np.mean([d[2] for d in dp_original]), np.mean([d[2] for d in dp_generated]))
                print(buf)
                print2file(buf, outFile)

                # Subgroup metrics for original data
                subgroups = ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)]
                data_pred_original = clf.predict(data_test_x)
                data_pred_gen = clf_generated.predict(data_test_x)
                subgroup_metrics_original = calculate_subgroup_metrics(data_test_x, data_test_y, data_pred_original, subgroups, S_start_index)
              
                log_subgroup_metrics(subgroup_metrics_original, "Original", outFile)

                # Log DP for each pair
                log_dp_metrics(dp_original, "Original", subgroups, S_start_index, outFile)
                log_dp_metrics(dp_generated, "Generated", subgroups, S_start_index, outFile)

                # Subgroup metrics for generated data
                subgroup_metrics_generated = calculate_subgroup_metrics(data_test_x, data_test_y, data_pred_gen, subgroups, S_start_index)
             
                log_subgroup_metrics(subgroup_metrics_generated, "Generated", outFile)

                # Save the models for each run and lambda
                if lamda == 0:
                    save_model(generator, f'generator_pre_fairness_new_run_{i}.pt')
                    save_model(critic, f'critic_pre_fairness_new_run_{i}.pt')
                else:
                    run_model_path = os.path.join(fairness_model_path, f'run_{i}_lambda_{lamda}')
                    os.makedirs(run_model_path, exist_ok=True)
                    save_model(generator, fairness_generator_path)
                    save_model(critic, fairness_critic_path) 