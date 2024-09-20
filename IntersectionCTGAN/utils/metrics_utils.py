#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from utils.seed_utils import set_random_seeds

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from models.ctgan_modified import CTGAN
from utils.model_utils import load_model, load_optimizer
from utils.data_sampler import DataSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from models.generator import Generator
from models.discriminator import Discriminator

# Define SUBGROUPS dictionary here
SUBGROUPS = {
    'Sex_age': ['Male_<25', 'Female_<25', 'Male_25–60', 'Female_25–60', 'Male_>60', 'Female_>60']
}


def calculate_dp_per_pair(data_x, data_y, indices_pairs):
    dp_per_pair = []
    for u_index, p_index in indices_pairs:
        underpriv_mask = data_x[:, u_index] == 1
        priv_mask = data_x[:, p_index] == 1

        if np.any(underpriv_mask) and np.any(priv_mask):
            underpriv_outcomes = data_y[underpriv_mask]
            priv_outcomes = data_y[priv_mask]
            underpriv_odds = np.mean(underpriv_outcomes)
            priv_odds = np.mean(priv_outcomes)
            disparity = priv_odds - underpriv_odds
            dp_per_pair.append(disparity)
        else:
            dp_per_pair.append(np.nan)  # Append NaN if there's insufficient data for a meaningful calculation

    return dp_per_pair

def calculate_metrics_per_subgroup(clf, data_x, data_y, subgroups):
    metrics = {}
    for group in subgroups:
        mask = data_x[:, SUBGROUPS['Sex_age'].index(group)] == 1
        if np.any(mask):
            subgroup_x = data_x[mask]
            subgroup_y = data_y[mask]
            if len(subgroup_y) > 0:
                pred_y = clf.predict(subgroup_x)
                accuracy = accuracy_score(subgroup_y, pred_y)
                f1 = f1_score(subgroup_y, pred_y)
                metrics[group] = {'accuracy': accuracy, 'f1': f1}
            else:
                metrics[group] = {'accuracy': np.nan, 'f1': np.nan}
        else:
            metrics[group] = {'accuracy': np.nan, 'f1': np.nan}
    return metrics


def calculate_demographic_parity(data_x, data_y, indices_pairs):
    disparities = []
    for u_index, p_index in indices_pairs:
        underpriv_mask = data_x[:, u_index] == 1
        priv_mask = data_x[:, p_index] == 1

        if np.any(underpriv_mask) and np.any(priv_mask):
            underpriv_outcomes = data_y[underpriv_mask]
            priv_outcomes = data_y[priv_mask]
            underpriv_odds = np.mean(underpriv_outcomes)
            priv_odds = np.mean(priv_outcomes)
            disparity = priv_odds - underpriv_odds
            disparities.append(disparity)
        else:
            disparities.append(np.nan)  # Append NaN if there's insufficient data for a meaningful calculation

    # Calculate the average of the non-NaN disparities
    valid_disparities = [d for d in disparities if not np.isnan(d)]
    if valid_disparities:
        return np.mean(valid_disparities)
    else:
        return np.nan  # Return NaN if all disparities are NaN

def evaluate_model(generator_path, discriminator_path, optimizerG_path, optimizerD_path, transformer, train_data, test_data, device):
    set_random_seeds(42)  # Ensure evaluation is reproducible
    ctgan = CTGAN(embedding_dim=128,
                  generator_dim=(256, 256),
                  discriminator_dim=(256, 256),
                  batch_size=500,
                  epochs=1,  # Minimal epochs to initialize components
                  apply_fairness_constraint=False,
                  cuda=True)
    ctgan._transformer = transformer
    transformed_data = transformer.transform(train_data)
    ctgan._data_sampler = DataSampler(transformed_data, transformer.output_info_list, ctgan._log_frequency)
    ctgan._generator = Generator(
        ctgan._embedding_dim + ctgan._data_sampler.dim_cond_vec(),
        ctgan._generator_dim,
        transformer.output_dimensions
    ).to(ctgan._device)
    ctgan._discriminator = Discriminator(
        transformer.output_dimensions + ctgan._data_sampler.dim_cond_vec(),
        ctgan._discriminator_dim,
        pac=ctgan.pac
    ).to(ctgan._device)
    load_model(ctgan._generator, generator_path)
    load_model(ctgan._discriminator, discriminator_path)

    optimizerG = torch.optim.Adam(ctgan._generator.parameters())
    optimizerD = torch.optim.Adam(ctgan._discriminator.parameters())
    load_optimizer(optimizerG, optimizerG_path, ctgan._device)
    load_optimizer(optimizerD, optimizerD_path, ctgan._device)

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

    clf = DecisionTreeClassifier()
    clf.fit(data_train_x, np.argmax(data_train_y, axis=1))
    accuracy_original = accuracy_score(clf.predict(data_test_x), np.argmax(data_test_y, axis=1))
    f1_original = f1_score(clf.predict(data_test_x), np.argmax(data_test_y, axis=1))

    clf_generated = DecisionTreeClassifier()
    clf_generated.fit(generated_x, np.argmax(generated_y, axis=1))
    accuracy_generated = accuracy_score(clf_generated.predict(data_test_x), np.argmax(data_test_y, axis=1))
    f1_generated = f1_score(clf_generated.predict(data_test_x), np.argmax(data_test_y, axis=1))

    indices_pairs = [(130, 131), (133, 132), (135, 134)]
    demographic_parity_original = calculate_demographic_parity(data_train_x, np.argmax(data_train_y, axis=1), indices_pairs)
    demographic_parity_generated = calculate_demographic_parity(generated_x, np.argmax(generated_y, axis=1), indices_pairs)
    demographic_parity_diff = abs(demographic_parity_original - demographic_parity_generated)

    dp_pairs_original = calculate_dp_per_pair(data_train_x, np.argmax(data_train_y, axis=1), indices_pairs)
    dp_pairs_generated = calculate_dp_per_pair(generated_x, np.argmax(generated_y, axis=1), indices_pairs)

    # Calculate metrics for each subgroup
    original_metrics = calculate_metrics_per_subgroup(clf, data_test_x, np.argmax(data_test_y, axis=1), SUBGROUPS['Sex_age'])
    generated_metrics = calculate_metrics_per_subgroup(clf_generated, data_test_x, np.argmax(data_test_y, axis=1), SUBGROUPS['Sex_age'])

    return accuracy_original, accuracy_generated, f1_original, f1_generated, demographic_parity_original, demographic_parity_generated, demographic_parity_diff, original_metrics, generated_metrics, dp_pairs_original, dp_pairs_generated


