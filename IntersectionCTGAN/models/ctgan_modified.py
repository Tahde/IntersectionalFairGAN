#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import torch.nn.functional as F

import warnings
import torch
import pandas as pd
from torch import optim
from tqdm import tqdm
from utils.model_utils import save_model, load_model, save_optimizer, load_optimizer
from models.base_module import BaseSynthesizer, random_state
from utils.data_sampler import DataSampler
from utils.data_transformer import DataTransformer

class CTGAN(BaseSynthesizer):
    def __init__(self, lambda_fairness=0.1, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=0.0002, generator_decay=1e-6, discriminator_lr=0.0002, discriminator_decay=1e-6,
                 batch_size=500, discriminator_steps=1, log_frequency=True, verbose=False, epochs=300, pac=1, cuda=True,
                 apply_fairness_constraint=True, sensitive_attribute_index=None, target_attribute_index=None, Y_start_index=None, desire_index=None,
                 underpriv_indices=None, priv_indices=None):
        assert batch_size % 2 == 0
        self.lambda_fairness = lambda_fairness
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.apply_fairness_constraint = apply_fairness_constraint
        self.sensitive_attribute_index = sensitive_attribute_index
        self.target_attribute_index = target_attribute_index
        self.Y_start_index = Y_start_index
        self.desire_index = desire_index
        self.underpriv_indices = underpriv_indices
        self.priv_indices = priv_indices

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'
        #print(f"Using device: {device}")
        self._device = torch.device(device)
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss', 'Demographic Parity'])

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        for _ in range(10):
            transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed
        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        return torch.cat(data_t, dim=1)

    def calculate_fairness_loss(self, generator_output, priv_indices, underpriv_indices):
        Y_start_index = self.Y_start_index
        desire_index = self.desire_index
    
        fairness_loss = 0.0
        for underpriv_index, priv_index in zip(underpriv_indices, priv_indices):
            underpriv_mask = generator_output[:, underpriv_index] == 1
            priv_mask = generator_output[:, priv_index] == 1
    
            outcome_index = Y_start_index + desire_index
            if outcome_index >= generator_output.shape[1]:
                raise IndexError(f"Outcome index {outcome_index} exceeds the bounds of the generator output with shape {generator_output.shape}")
    
            underpriv_positive_outcomes = generator_output[underpriv_mask, outcome_index]
            priv_positive_outcomes = generator_output[priv_mask, outcome_index]
    
            if len(underpriv_positive_outcomes) > 0 and len(priv_positive_outcomes) > 0:
                underpriv_rate = torch.mean(underpriv_positive_outcomes)
                priv_rate = torch.mean(priv_positive_outcomes)
                disparity = torch.abs(priv_rate - underpriv_rate)
                fairness_loss += disparity
            else:
                disparity = torch.tensor(float('nan'))  # If there are no samples, return NaN
    
        return fairness_loss

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c
        loss = torch.stack(loss, dim=1)
        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')
        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def fit(self, train_data, discrete_columns=(), epochs=None, fairness_epochs=0, optimizerG=None, optimizerD=None):
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss', 'Demographic Parity'])
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        if self._generator is None:
            self._generator = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim
            ).to(self._device)

        if self._discriminator is None:
            self._discriminator = Discriminator(
                data_dim + self._data_sampler.dim_cond_vec(),
                self._discriminator_dim,
                pac=self.pac
            ).to(self._device)

        if optimizerG is None:
            optimizerG = optim.Adam(
                self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
                weight_decay=self._generator_decay
            )

        if optimizerD is None:
            optimizerD = optim.Adam(
                self._discriminator.parameters(), lr=self._discriminator_lr,
                betas=(0.5, 0.9), weight_decay=self._discriminator_decay
            )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        epoch_iterator = tqdm(range(1, epochs + fairness_epochs + 1), disable=(not self._verbose))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        for epoch in epoch_iterator:
            apply_fairness = (epoch > epochs) and (self.lambda_fairness > 0)

            epoch_gen_loss = 0
            epoch_disc_loss = 0
            epoch_fairness_loss = 0

            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                    epoch_disc_loss += loss_d.item()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                # Debugging inside the fit method after generating the fakeact
                # print("Shape of generator output:", fakeact.shape)
                # generator_output_sample = fakeact.detach().cpu().numpy()[:5]
                # print("Sample generator output:", generator_output_sample)
                # max_index = fakeact.shape[1] - 1
                # print(f"Maximum index for generator output: {max_index}")

                # underpriv_index = self.underpriv_indices[0]
                # priv_index = self.priv_indices[0]
                # Y_start_index = self.Y_start_index
                # desire_index = self.desire_index
                # outcome_indices = list(range(Y_start_index, Y_start_index + 2))

                #print(f"Underprivileged index: {underpriv_index}, Privileged index: {priv_index}")
                # print(f"Outcome indices: {outcome_indices}")

                # if underpriv_index > max_index or priv_index > max_index or any(idx > max_index for idx in outcome_indices):
                #     raise IndexError(f"One of the indices exceeds the bounds of the generator output. Max index: {max_index}")

                # print("Underprivileged positive outcomes:", generator_output_sample[generator_output_sample[:, underpriv_index] == 1][:, outcome_indices])
                # print("Privileged positive outcomes:", generator_output_sample[generator_output_sample[:, priv_index] == 1][:, outcome_indices])

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                if apply_fairness and self.apply_fairness_constraint:
                    fairness_loss = self.calculate_fairness_loss(
                        fakeact,
                        self.priv_indices,
                        self.underpriv_indices)
                    loss_g += self.lambda_fairness * fairness_loss
                    epoch_fairness_loss += fairness_loss.item()

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

                epoch_gen_loss += loss_g.item()

            epoch_gen_loss /= steps_per_epoch
            epoch_disc_loss /= steps_per_epoch
            epoch_fairness_loss /= steps_per_epoch

            print(f"Epoch {epoch}: Generator loss = {epoch_gen_loss}, Discriminator loss = {epoch_disc_loss}, Fairness loss = {epoch_fairness_loss}")
            new_row = pd.DataFrame([{'Epoch': epoch, 'Generator Loss': epoch_gen_loss,
                         'Discriminator Loss': epoch_disc_loss, 'Demographic Parity': epoch_fairness_loss}])
            self.loss_values = pd.concat([self.loss_values, new_row], ignore_index=True)

            # Save general models at the end of the general training phase
            #if epoch == epochs:
                #print(f"Saving models at epoch {epoch} before fairness training.")
                #save_model(self._generator, 'generator_pre_fairness.pt')
                #save_model(self._discriminator, 'discriminator_pre_fairness.pt')
                #save_optimizer(optimizerG, 'optimizerG_pre_fairness.pt')
                #save_optimizer(optimizerD, 'optimizerD_pre_fairness.pt')

        # Debugging indices of all subgroups
        # print(f"Shape of generator output: {fakeact.shape}")
        # generator_output_sample = fakeact.detach().cpu().numpy()[:5]
        # print("Sample generator output:", generator_output_sample)
        # max_index = fakeact.shape[1] - 1
        # print(f"Maximum index for generator output: {max_index}")

        for underpriv_index, priv_index in zip(self.underpriv_indices, self.priv_indices):
            print(f"Underprivileged index: {underpriv_index}, Privileged index: {priv_index}")

        print(f"Outcome indices: {self.Y_start_index} to {self.Y_start_index + 1}")
        print("Outcome indices:", list(range(self.Y_start_index, self.Y_start_index + 2)))

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
        if self._discriminator is not None:
            self._discriminator.to(self._device)

