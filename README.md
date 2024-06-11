## Project Overview
This project demonstrates the process of preparing data, registering assets, training a model, and deploying it using Azure Machine Learning. The entire workflow is executed in Python sdk v2.

## Steps to Recreate the Project

### 1. Preparing the Data
The dataset consists of images of cards, which needed to be unzipped and prepared for training.

### 2. Registering the Data Asset
After preparing the data, it was registered as a data asset in Azure ML.

### 3. Registering the Environment
Environment was created and registered for training the model.

### 4. Performing Training with Command Job
Created a training script (`train.py`) and run it as a command job in Azure ML.

#### `train.py` Example:
The `train.py` script includes the necessary code to train the model.

### 5. Registering the Model
After training, registered the trained model in Azure ML for deployment.

### 6. Registering and Deploying the Endpoint
Create and register an endpoint in Azure ML and deploy the model along with the environment. Script (`score.py`) was created for custom deployment

### 7. Testing the Deployed Model
Invoke the deployed model to test its performance using the Azure ML deployment website.
