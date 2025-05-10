# Smart Home Power Consumption Prediction

This repository contains various deep learning models for predicting power consumption in a smart home using weather information. The models implemented include Bi-directional LSTM, CRNN, LSTM, and RGAN.

## Dataset

The dataset used for training and evaluation can be found on Kaggle: [Smart Home Dataset with Weather Information](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information).

## Project Structure


## Notebooks

### Bi-directional-LSTM_training.ipynb

This notebook implements a Bi-directional LSTM model for predicting power consumption. The model is trained using the following steps:
1. **Data Preprocessing**: Cleaning and normalizing the data.
2. **Model Definition**: Creating a Bi-directional LSTM model with dropout layers.
3. **Training**: Using early stopping and learning rate reduction callbacks.
4. **Evaluation**: Calculating Mean Absolute Error (MAE) and plotting training and validation loss.

### CRNN_training.ipynb

This notebook implements a Convolutional Recurrent Neural Network (CRNN) model for predicting power consumption. The model is trained using the following steps:
1. **Data Preprocessing**: Cleaning and normalizing the data.
2. **Model Definition**: Creating a CRNN model with convolutional, LSTM, and dense layers.
3. **Training**: Using early stopping and learning rate reduction callbacks.
4. **Evaluation**: Calculating Mean Absolute Error (MAE) and plotting training and validation loss.

### LSTM_training.ipynb

This notebook implements an LSTM model for predicting power consumption. The model is trained using the following steps:
1. **Data Preprocessing**: Cleaning and normalizing the data.
2. **Model Definition**: Creating an LSTM model with multiple LSTM layers and a dense output layer.
3. **Training**: Using early stopping and learning rate reduction callbacks.
4. **Evaluation**: Calculating Mean Absolute Error (MAE) and plotting training and validation loss.

### RGAN_training.ipynb

This notebook implements a Recurrent Generative Adversarial Network (RGAN) model for predicting power consumption. The model is trained using the following steps:
1. **Data Preprocessing**: Cleaning and normalizing the data.
2. **Model Definition**: Creating generator and discriminator models using LSTM layers.
3. **Training**: Using a custom training loop for the GAN.
4. **Evaluation**: Calculating Mean Absolute Error (MAE) and plotting training and validation loss.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Matplotlib

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/balamanikantsai/Smart_home_power_consumption_prediction.git
   cd smart_home_power_consumption_prediction
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the notebooks: Open any of the Jupyter notebooks (.ipynb files) in Jupyter Notebook or Jupyter Lab and run the cells to train and evaluate the models.

## Results

The results of the models are evaluated using Mean Absolute Error (MAE) and visualized using training and validation loss plots. Each notebook contains detailed results and plots for the respective model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.