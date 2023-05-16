# Credit Card Fraud Prediction Readme
Throughout this project I created a tensorflow neural network to make a binary classification prediction on credit card transaction.




# Data Processing
## Loading the libraries and dataset
```python
import tensorflow as tf
# Needed for MaxPool
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
# Needed for dataframes
import pandas as pd
#needed to translate labels into numpy
import numpy as np
# Needed for plotting the data
import matplotlib.pyplot as plt
# Needed for splitting the dataframe
from sklearn.model_selection import train_test_split
# Needed for scaling the features
from sklearn.preprocessing import StandardScaler

credit_card_df = pd.read_csv("https://query.data.world/s/gyyiyxdr6fpnx42ragtni6ggdz5pok?dws=00000")
credit_card_df.head()
credit_card_df.shape
```
## Separating the transactions into positive and negative dataframes
```python
fraud_postitive_df = credit_card_df[credit_card_df["Class"]==1]
print(fraud_postitive_df.head())
fraud_negitive_df = credit_card_df[credit_card_df["Class"]==0]
print(fraud_negitive_df.head())
fraud_postitive_df.shape , fraud_negitive_df.shape

```

## Reshaping and combining the datasets
```python
# Resizing the non fraud to be the same size as the fraud
fraud_negitive_df = fraud_negitive_df.head(492)
fraud_negitive_df.shape

# Combining the two datasets
fraud_testing_df = fraud_negitive_df.append(fraud_postitive_df, ignore_index = True)
fraud_testing_df
```

## Splitting the dataset into features and labels
```python
fraud_testing_labels = fraud_testing_df["Class"]
print(fraud_testing_labels)
fraud_testing_features = fraud_testing_df.drop("Class", axis = 1)
print(fraud_testing_features)
```

# Test Train Split
## Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(fraud_testing_features, fraud_testing_labels, test_size = 0.2, random_state = 0, stratify = fraud_testing_labels)
```

## Scaling the features
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print(f"x train {X_train.shape}")

print(f" y test {y_test.shape}")
```

## Creating a 3 dimension dataset
```python
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

X_train.shape, X_test.shape
```

# Model
## Model Architecture
```python
epochs = 10
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

```
## Model Compiling
```python
model.compile(optimizer= 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])

```
## Model Training
```python
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

```
## Model Results
#### Getting evaluation metrics
```python
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (accuracy*100))

from sklearn.metrics import precision_recall_fscore_support

y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred)
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred_classes)
print('Precision:', precision)
print('Recall:', recall)

from sklearn.metrics import f1_score

y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred)
f1score = f1_score(y_test, y_pred_classes)
print('F1-score:', f1score)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred)
cm = confusion_matrix(y_test, y_pred_classes)
print('Confusion Matrix:\n', cm)
```
![Screenshot 2023-05-01 at 8 15 17 PM](https://user-images.githubusercontent.com/123593094/235558844-417b0b91-26a9-47bf-9f0a-a84203a741af.png)
#### Seeing the feature importance
```python 
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# Create a separate variable to store the original 2D shape of X_test before reshaping
X_test_original = X_test.reshape(X_test.shape[0], X_test.shape[1])

def model_wrapper(X):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return model.predict(X).flatten()

def custom_scorer(model_func, X, y):
    y_pred = np.round(model_func(X))
    return accuracy_score(y, y_pred)

# Pass X_test_original (2D) and custom_scorer to the permutation_importance function
result = permutation_importance(model_wrapper, X_test_original, y_test, n_repeats=10, random_state=0, n_jobs=-1, scoring=custom_scorer)

for i in range(X_train.shape[1]):
    print(f'Feature {i + 1}: {result.importances_mean[i]:.5f} +/- {result.importances_std[i]:.5f}')

```
![Screenshot 2023-05-01 at 8 16 41 PM](https://user-images.githubusercontent.com/123593094/235558978-3a9897ec-8d6d-4780-9859-6308d3148087.png)

#### Creating `plot_learningCurve()`
```python
def plot_learningCurve(history, epoch):
  # Plot training & validation accuracy values
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()
```

#### Plotting learning curve
```python 
plot_learningCurve(history, epochs)
```
![image](https://user-images.githubusercontent.com/123593094/235556428-e556a683-df2a-4428-a6e0-da3879f0a280.png)
![image](https://user-images.githubusercontent.com/123593094/235556439-0e6a4d2c-135e-4e3f-a20a-9be55ec1b649.png)

# Second Model

## Model Changes
```python
epochs = 50
model.add(MaxPool1D(2))
```
## Model Training

## Model Results
#### Evaluation metrics
![Screenshot 2023-05-01 at 8 17 36 PM](https://user-images.githubusercontent.com/123593094/235559061-4862b545-58c9-4f9e-8be5-26e74d8b65b8.png)

#### Feature Importance
![Screenshot 2023-05-01 at 8 18 16 PM](https://user-images.githubusercontent.com/123593094/235559131-371ea3cc-a54f-462d-a57e-3acdd99de31d.png)

#### Learning curve

![image](https://user-images.githubusercontent.com/123593094/235557945-f2a278f9-7781-4e17-9371-5a809ba27323.png)
![image](https://user-images.githubusercontent.com/123593094/235557955-7ae7d8a8-3cde-491e-9839-11aa47c61d4e.png)


