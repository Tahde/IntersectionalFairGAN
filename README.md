# IntersectionalFairGAN
## Description

This repository implements **IntersectionalFairGAN (IFGAN)**, a tool for addressing fairness in machine learning models by considering the intersectionality of sensitive attributes. These codes are based on the paper _"Enhancing Tabular GAN Fairness: The Impact of Intersectional Feature Selection."_ 

It builds upon two state-of-the-art GAN models for tabular data generation, **[TabFairGAN](https://github.com/amirarsalan90/TabFairGAN)** and **[CTGAN](https://github.com/sdv-dev/CTGAN)**, modifying the loss function to include an intersectional demographic parity constraint.

- **[TabFairGAN](https://github.com/amirarsalan90/TabFairGAN)**: We extended this model to handle intersectionality by modifying the loss function, focusing on two sensitive features (e.g., Gender-Age in the Adult dataset).
  
- **[CTGAN](https://github.com/sdv-dev/CTGAN)**: We modified CTGAN's generator loss function to incorporate an intersectional demographic parity constraint, addressing fairness across multiple sensitive attributes.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Tahde/IntersectionalFairGAN.git
   
## Dependencies
Ensure you have the following installed:
- **Python 3.7+**
- **PyTorch**
- **Pandas**
- **Scikit-learn**

### Running the Project

1. **Train the Model**:
   To train the IFGAN model for TabFairGAN, run:

   ```bash
   python IntersectionTabFair/scripts/train_IntersectionFairGAN.py

   ```

   The training consists of two phases:
   - **Phase I**: Trains the generator without fairness constraints (λf = 0), and the best model is selected based on the lowest sum of differences in **accuracy**, **F1 score**, and **demographic parity** compared to the original data. The general models are saved in the `general_models` folder.
   - **Phase II**: Applies an intersectional demographic parity constraint. The best model is selected based on the **highest absolute difference in demographic parity**, while also considering the **lowest F1 difference** and **lowest accuracy difference**. The fairness models are saved in the `fairness_models` folder.

   The training is repeated 10 times for each λf, and the **best models** are manually placed in the `best_models` folder for CSV generation.

2. **Generate CSV Files**:
   After placing the best models in the `best_models` folder, generate CSV files using:

   ```bash
   python Intersectional-TabFair-Adult-Csv-generator.py
   ```
Yes, adding a **Citations** section is a good practice, especially if your repository is based on academic work or uses models and methods from other research papers. Here’s how you can incorporate a **Citations** section into your README:

---

## Citations
If you use this code, please cite the following:
   ```
   @inproceedings{dehdarirad2024intersectionalfairgan,
       title={Enhancing Tabular GAN Fairness: The Impact of Intersectional Feature Selection},
       author={Tahereh Dehdarirad and Ericka Johnson and Gabriel Eilertsen and Saghi Hajisharif},
       booktitle={Proceedings of the 23rd International Conference on Machine Learning and Applications (ICMLA 2024)},
       year={2024}
   }
   ```
