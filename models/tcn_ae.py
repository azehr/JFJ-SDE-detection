"""
Title: tcn_ae.py

Description:
    Implements a temporal autoencoder. A linear classifier is placed on the latent
    space of the autoencoder. The autoencoder and classifier are trained in 
    unison using a weighted loss of both the classification loss and reconstruction
    loss. This enables the autoencoder to learn a discriminative latent space on 
    the data.
    
Author: Andrew Zehr

"""

# Load needed packages
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from utils.data_handling import read_JFJ_data
from utils.model_evaluation import (
    evaluate_model, smooth_predictions, visualize_feature_separation)
from utils.plotting import plot_predictions
from utils.calibration import plot_calibration_curve

# Add kernel to penealize model switching between states in the predictions
# This has the effect of smoothing the predictions
transition_penalty_kernel = (torch.tensor([[-1, 1]], dtype=torch.float32)).view(1,1,-1)


class TemporalAutoencoder(torch.nn.Module):
    """ Define the autoencoder """
    def __init__(self, input_dim, latent_dim, conv_dim, kernel_size = 3):
        super().__init__()
        self.input_dim = input_dim
        self.conv_dim = conv_dim
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.pad = int((self.kernel_size-1)/2)
        
        self.conv1 = nn.Conv1d(self.input_dim, self.conv_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(self.conv_dim)

        
        self.conv2 = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.kernel_size, padding=self.pad)
        self.bn2 = nn.BatchNorm1d(self.conv_dim)
        
        self.conv3 = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.kernel_size, padding=self.pad)
        self.bn3 = nn.BatchNorm1d(self.conv_dim)
        self.relu = nn.ReLU()
        


        
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        self.encoder = nn.Sequential(
            nn.Linear(self.conv_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )
        
        
        
        
        self.inv_conv = nn.ConvTranspose1d(self.input_dim, self.input_dim, kernel_size=self.kernel_size, padding=self.pad)
 
    def forward(self, x):
        x = torch.transpose(x, 0,1)
        x = torch.transpose(self.conv1(x), 0, 1)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = torch.transpose(x, 0,1)
        x = torch.transpose(self.conv2(x), 0, 1)
        x = self.bn2(x)
        x = self.relu(x)
        x = torch.transpose(x, 0,1)
        x = torch.transpose(self.conv3(x), 0, 1)
        x = self.bn3(x)
        convolved = self.relu(x)
        
        
        encoded = self.encoder(convolved)
        decoded = self.decoder(encoded)
        reconstruction = torch.transpose(self.inv_conv(torch.transpose(decoded,0,1)),0,1)
        return encoded, reconstruction



class Classifier(torch.nn.Module):
    """ Define the linear classifer """
    def __init__(self, latent_dim):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 1),
            )
        
        self.transition_penalty_conv = nn.Conv1d(
            in_channels=1,  # Number of classes in your output
            out_channels=1,           # Single channel output
            kernel_size=2,            # Convolutional kernel size (captures pairs)
            stride=1,
            padding=0,
            bias=False,
        )
        
        self.make_probs = nn.Sigmoid()
        
    def forward(self, x):
        prob = self.classifier(x)
        
        return torch.FloatTensor(prob)


class AELC(nn.Module):
    """ Combine the autoencoder and linear classifiern"""
    def __init__(self, autoencoder, classifier):
        super().__init__()
        
        self.autoencoder = autoencoder
        self.classifier = classifier
        
    def forward(self, x):
        encoded, reconstructed = self.autoencoder(x)
        classification = self.classifier(encoded)
        return classification, reconstructed, encoded
    

class WeightedLoss(nn.Module):
    """ Define weighted loss with reconstruction and classification loss """
    def __init__(self, recon_weight, class_weight):
        super().__init__()
        self.recon_weight = recon_weight
        self.class_weight = class_weight

    def forward(self, reconstruct, pred, x_true, labels):
        mse = nn.MSELoss()
        ce = nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
        reconstruction_loss = mse(reconstruct, x_true) 
        classification_loss = ce(pred, labels)
        
        loss = (self.recon_weight * reconstruction_loss 
                + ((1-self.recon_weight) * classification_loss))
        return loss, reconstruction_loss.item(), classification_loss.item()
    

class CustomDataset(Dataset):
    def __init__(self, X, y):
        """Initializes instance of class CustomDataSet."""
        
        # Save target and predictors
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [
            torch.tensor(self.X.iloc[idx], dtype = torch.float32),
            torch.tensor(self.y[idx], dtype = torch.float32),
            ]
    

def fit_model(model, epochs, trainLoader, weight):
    """ Fit the model and plot the loss curves"""
    np.random.seed(1)
    losses = []
    class_losses = []
    recon_losses = []
    for epoch in range(epochs):
        for X, y in trainLoader:
          loss_batch = 0
          recon_batch = 0
          class_batch = 0
          # Output of Autoencoder
          y = y.unsqueeze(1)
          y_pred, reconstructed, latent = model(X)
           

          # Calculating the loss function
          loss, recon_loss, class_loss = loss_function(reconstructed, y_pred, X, y)
           
          # The gradients are set to zero,
          # the gradient is computed and stored.
          # .step() performs parameter update
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          loss_batch += loss.item()
          recon_batch += recon_loss
          class_batch += class_loss
          
        # Storing the losses in a list for plotting
        if epoch % 10 == 0:
            print("Epoch {:03d}: Loss: {:.3f}  Recon. Loss: {:.3f}  Class. Loss: {:.3f}".format(epoch, loss_batch, recon_batch, class_batch))
        losses.append(loss_batch)
        class_losses.append(class_batch)
        recon_losses.append(recon_batch)

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title(f"Weighted Loss (Lambda = {weight})", fontsize=18)
    
    plt.subplot(2, 2, 3)
    plt.plot(recon_losses)
    plt.title(f"Reconstruction ({weight})", fontsize=12)
    
    plt.subplot(2, 2, 4)
    plt.plot(class_losses)
    plt.title(f"Classification ({1-weight})", fontsize=12)
    plt.tight_layout()
    plt.show()
    
  

""" Load the data, labels, and splits """
allData = read_JFJ_data(
    r"\processed\iterative_impute_df.csv"
    ).dropna()
allData["neg_AE_SSA"] = (allData.AE_SSA < 0).astype(int)

data = read_JFJ_data(r"\processed\imputed2020.csv")



dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
Y = dust_event_info.sde_event

splits = pd.read_json("six_train_test_splits.json")
plotting_splits = pd.read_json("plotting_splits.json")



""" Experiment with the model"""
latent_dim = 10
reconstruction_weight = 0.5
reg = 1e-4
learn_rate = 1.5e-4
epoch_num = 35
balance_mult = 1
conv_dim = 512
balance_classes = False

# Use the model for creating a plot
predictions = pd.Series(dtype=int)
predicted_probs = pd.Series(dtype=float)
for i, row in plotting_splits.iterrows():
    np.random.seed(1)
    
    x_train = data[data.block_nr.isin(row["trainSet"])].drop("block_nr", axis = 1)
    x_test = data[data.block_nr.isin(row["testSet"])].drop("block_nr", axis = 1)

    Y_train = Y[x_train.index]
    Y_test = Y[x_test.index]
    
    sc = StandardScaler()
    train_scale = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
    test_scale = pd.DataFrame(sc.transform(x_test), columns = x_test.columns, index = x_test.index)
    trainDataset = CustomDataset(train_scale, Y_train)
    testDataset = CustomDataset(test_scale, Y_test)
 
    # batch size of 1 week
    train_dataloader = DataLoader(trainDataset, batch_size = 7 * 24, shuffle = False)
    
    input_dim = x_train.shape[1]

    # Model Initialization
    #state_dict = torch.load('autoencoder_all_data.pth')    
    ae = TemporalAutoencoder(input_dim, latent_dim, conv_dim, kernel_size=3)
    #ae.load_state_dict(state_dict)
    cl = Classifier(latent_dim)
    model = AELC(ae, cl)

    # Using the Weighted Loss between reconstruction loss and classification loss
    if balance_classes:
        class_weight = torch.FloatTensor([balance_mult * (1-Y_train.mean())/Y_train.mean()])
    else:
        class_weight = torch.FloatTensor([1])

    loss_function = WeightedLoss(recon_weight = reconstruction_weight, class_weight=class_weight)
 
    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate, weight_decay= reg)
    

    print(f"Start fitting model {i}")  
    fit_model(model, epochs = epoch_num, trainLoader = train_dataloader, weight = reconstruction_weight)
    print(f"Done fitting model {i}")
    
    
    test_tensor = torch.from_numpy(test_scale.to_numpy()).to(torch.float32)
    
    # Evaluate Model
    model.eval()
    f = nn.Sigmoid()
    probs = f(model(test_tensor)[0]).detach().numpy().reshape(-1)
    pred = (probs >= 0.5).astype(int)
    
    
    print(f"Balanced Accuracy of model {i} is {balanced_accuracy_score(Y_test, pred)}")
    
    predictions = pd.concat([predictions, pd.Series(pred, index = x_test.index)], axis = 0)
    predicted_probs = pd.concat([predicted_probs, pd.Series(probs, index = x_test.index)], axis=0)

pred_smooth = smooth_predictions(predictions).apply(lambda x: 1 if x > 0.5 else 0)
evaluate_model(predictions, dust_event_info)
plot_predictions(pd.DataFrame({"Predictions": pred_smooth}, index = data.index), dust_event_info)

plot_calibration_curve(dust_event_info.loc[predicted_probs.index].sde_event, predicted_probs)


model(test_tensor)[2]




# Train the final model and use it to reduce the dimension of the data
latent_dim = 10
reconstruction_weight = 0.2
reg = 1e-4
learn_rate = 1.5e-4
epoch_num = 80
balance_mult = 1
conv_dim = 512
balance_classes = False

np.random.seed(1)

x_train = data.drop("block_nr", axis = 1)
x_test = allData

Y_train = Y[x_train.index]

sc = StandardScaler()
train_scale = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
test_scale = pd.DataFrame(sc.transform(x_test), columns = x_test.columns, index = x_test.index)
trainDataset = CustomDataset(train_scale, Y_train)
testDataset = CustomDataset(test_scale, Y_test)

# batch size of 1 week
train_dataloader = DataLoader(trainDataset, batch_size = 7 * 24, shuffle = False)

input_dim = x_train.shape[1]

# Model Initialization
#state_dict = torch.load('autoencoder_all_data.pth')    
ae = TemporalAutoencoder(input_dim, latent_dim, conv_dim, kernel_size=3)
#ae.load_state_dict(state_dict)
cl = Classifier(latent_dim)
model = AELC(ae, cl)

# Using the Weighted Loss between reconstruction loss and classification loss
if balance_classes:
    class_weight = torch.FloatTensor([balance_mult * (1-Y_train.mean())/Y_train.mean()])
else:
    class_weight = torch.FloatTensor([1])

loss_function = WeightedLoss(recon_weight = reconstruction_weight, class_weight=class_weight)

# Using an Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate, weight_decay= reg)
 
fit_model(model, epochs = epoch_num, trainLoader = train_dataloader, weight = reconstruction_weight)


# Save Model
torch.save(model.state_dict(), "ae_tcn.pth")

# Transform all the data and save it
test_tensor = torch.from_numpy(test_scale.to_numpy()).to(torch.float32)

transformed_data = pd.DataFrame(model(test_tensor)[2].detach().numpy(), index = allData.index)

""" Uncomment this if you want to save the data """
# transformed_data.to_csv("tcn_ae_data.csv")

visualize_feature_separation(transformed_data, dust_event_info, file_name = "tcn_ae_separation.pdf")
