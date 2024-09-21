# IntersectionalFairGAN
## Description

This repository implements **IntersectionalFairGAN (IFGAN)**, a tool for addressing fairness in machine learning models by considering the intersectionality of sensitive attributes. These codes are based on the paper _"Enhancing Tabular GAN Fairness: The Impact of Intersectional Feature Selection."_ 

It builds upon two state-of-the-art GAN models for tabular data generation, **[TabFairGAN](https://github.com/amirarsalan90/TabFairGAN)** and **[CTGAN](https://github.com/sdv-dev/CTGAN)**, modifying the loss function to include an intersectional demographic parity constraint.

- **[TabFairGAN](https://github.com/amirarsalan90/TabFairGAN)**: We extended this model to handle intersectionality by modifying the loss function, focusing on two sensitive features (e.g., Gender-Age in the Adult dataset).
  
- **[CTGAN](https://github.com/sdv-dev/CTGAN)**: We modified CTGAN's generator loss function to incorporate an intersectional demographic parity constraint, addressing fairness across multiple sensitive attributes.

### How to Run the Models

This project contains two models, each located in its own folder. Follow the steps below to install dependencies and run each model.

#### Folder Structure
- `IntersectionTabFair/`: Contains the implementation of Intersectional TabFairGAN.
- `IntersectionCTGAN/`: Contains the implementation of Intersectional Fair CTGAN.
- Both models share the same data set, Adult.csv.

#### Installing Dependencies
Both models require the same dependencies. You can install them by running the following command from the root directory:
```bash
pip install -r requirements.txt
```
#### Training Phases for Both Models
The training for each model consists of two phases:
- **Phase I**: Trains the generator without fairness constraints (λf = 0), selecting the best model based on the lowest sum of differences in **accuracy**, **F1 score**, and **demographic parity** compared to the original data. The general models are saved in the `general_models` folder.
  
- **Phase II**: Adds an intersectional demographic parity constraint. The best model is selected based on the **highest absolute difference in demographic parity**, while minimizing differences in **F1 score** and **accuracy**. The fairness models are saved in the `fairness_models` folder.

Training is repeated 10 times for each λf, and the best models are manually placed in the `best_models` folder for generating CSV files.

#### Running Intersectional TabFairGAN 
1. Navigate to the `IntersectionTabFair` folder:
   ```bash
   cd IntersectionTabFair
   ```
2. Run the model training script:
   ```bash
   python scripts/train_IntersectionFairGAN.py

   ```
3. The dataset will be loaded from the `data/` folder, and results will be saved in `TabFairGAN/results/`.

#### Running Intersectional Fair CTGAN (IFCTGAN)
1. Navigate to the `CTGAN` folder:
   ```bash
   cd IntersectionCTGAN
   ```
2. Run the model training script:
   ```bash
   python main.py
   ```


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


2. **Generate CSV Files**:
   After placing the best models in the `best_models` folder, generate CSV files using:

   ```bash
   python Intersectional-TabFair-Adult-Csv-generator.py
   ```

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




