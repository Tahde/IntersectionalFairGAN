#!/usr/bin/env python
# coding: utf-8

# In[ ]:


if __name__ == '__main__':
    # Specify the dataset path and run the training pipeline
    generator, critic, accuracy, f1 = run_training_pipeline('adult.csv', batch_size=64, epochs=100, device='cuda')

