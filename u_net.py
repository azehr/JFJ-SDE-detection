"""
   Title: U_net
   
   Description: https://arxiv.org/pdf/1910.11162.pdf
                Inspired by this paper on using a U-Net architecture 
                for time series segmentation.
                
                Use the architecture to model the dust events
                
   Author: Andrew Zehr
   
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from utils.data_handling import read_JFJ_data
from utils.model_evaluation import evaluate_model
from utils.plotting import plot_predictions

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
    

class conv_block(nn.Module):
    """ Defines the convolutional block, which is repeated several times in the architecture"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 0, 1)
        x = self.conv1(inputs)
        x = torch.transpose(x, 0, 1)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.transpose(x, 0, 1)
        x = self.conv2(x)
        x = torch.transpose(x, 0, 1)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool1d(2)
    def forward(self, inputs):
        x = self.conv(inputs)
        x_p = torch.transpose(x, 0, 1)
        p = self.pool(x_p)
        p = torch.transpose(p, 0, 1)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        inputs = torch.transpose(inputs, 0, 1)
        skip = torch.transpose(skip, 0, 1)
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=0)
        x = torch.transpose(x, 0, 1)
        x = self.conv(x)
        return x


class unet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(input_dim, 128)
        self.e2 = encoder_block(128, 256)
        self.e3 = encoder_block(256, 512)
        self.e4 = encoder_block(512, 1024)

        
        """ Latent Space """
        self.b = conv_block(1024, 2048)
        
        """ Decoder """
        self.d1 = decoder_block(2048, 1024)
        self.d2 = decoder_block(1024, 512)
        self.d3 = decoder_block(512, 256)
        self.d4 = decoder_block(256, 128)
       
        """ Classifier """
        self.outputs = nn.Conv1d(128, 1, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        # Pad the input so it is a multiple of 16 (2^4), that way the downsampling 
        # dimensions remain integers always
        remainder = inputs.shape[0] % 16
        pad_needed = 16 - remainder
        
        if remainder != 0:
            inputs = torch.cat([inputs, inputs[-pad_needed:, :]], axis = 0)
        
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
       
        """ Latent """
        b = self.b(p4)
        
        """ Decoder """
        # Includes the skip connection data from the encoder layer
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(torch.transpose(d4,0,1))
        
        # Removes the padded entries to ensure output_dim = input_dim!
        if remainder != 0:
            outputs = outputs[:,:-pad_needed]
        
        return outputs
    
    
def fit_model(model, epochs, trainLoader, pos_weight):
    np.random.seed(1)
    losses = []
    # BCE Loss with class-weighting parameter
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        for X, y in trainLoader:
          loss_batch = 0

          # Output of Autoencoder
          y = y.unsqueeze(1)
          targets = torch.transpose(y,0,1)
          outputs = model(X)

          # Calculating the loss function
          loss = criterion(outputs, targets)
           
          # The gradients are set to zero,
          # the gradient is computed and stored.
          # .step() performs parameter update
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          loss_batch += loss.item()

          
        # Storing the losses in a list for plotting
        if epoch % 10 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch, loss_batch))
        losses.append(loss_batch)


    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.plot(losses)
    plt.title("Training Loss", fontsize=18)
    plt.show()




"""Load the data, labels, and splits"""
data = read_JFJ_data(
    r"\processed\data_imputed_2020_splits.csv"
    )
data["neg_AE_SSA"] = (data["AE_SSA"] < 0).astype(int)

data2017 = read_JFJ_data(r"\final\cleaned_impute.csv", date_range = ["2017-01-01 00:00:00","2017-12-31 23:59:59"])
data2017["neg_AE_SSA"] = (data2017["AE_SSA"] < 0).astype(int)

model_results_2020 = pd.read_csv("predictions_2020", index_col=0)
model_results_2017 = pd.read_csv("predictions_2017", index_col=0)


dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
dust_event_info_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")
Y = dust_event_info.loc[data.index].sde_event
Y_2017 = dust_event_info_2017.loc[data2017.index].sde_event

splits = pd.read_json("six_train_test_splits.json")
plotting_splits = pd.read_json("plotting_splits.json")




# Hyperparameter Tuning
regularization = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
balance_mult = [0, 1, 2, 3]
results = pd.DataFrame(columns=regularization, index = balance_mult)
input_dim = data.shape[1]
learn_rate = 1.5e-4
epoch_num = 20
predictions = pd.Series(dtype=int)
predicted_probs = pd.Series(dtype=float)

for reg in regularization:
    for balance in balance_mult:
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
         
            # batch size of 4 days
            train_dataloader = DataLoader(trainDataset, batch_size = 7 * 24, shuffle = False)
            
            input_dim = x_train.shape[1]

            # Model Initialization
            model = unet(input_dim)

            # Using the Weighted Loss between reconstruction loss and classification loss
            if balance > 0:
                class_weight = torch.FloatTensor([balance * (1-Y_train.mean())/Y_train.mean()])
            else:
                class_weight = torch.FloatTensor([1])

         
            # Using an Adam Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate, weight_decay= reg)
            

            print(f"Start fitting model {i}")  
            fit_model(model, epochs = epoch_num, trainLoader = train_dataloader, pos_weight=class_weight)
            print(f"Done fitting model {i}")
            
            
            test_tensor = torch.from_numpy(test_scale.to_numpy()).to(torch.float32)
            
            # Evaluate Model
            model.eval()
            f = nn.Sigmoid()
            probs = f(model(test_tensor)[0]).detach().numpy().reshape(-1)

            pred = (f(model(test_tensor)[0]).detach().numpy() >= 0.5).astype(int).reshape(-1)
            
            
            print(f"Balanced Accuracy of model {i} is {balanced_accuracy_score(Y_test, pred)}")
            
            predictions = pd.concat([predictions, pd.Series(pred, index = x_test.index)], axis = 0)
            predicted_probs = pd.concat([predicted_probs, pd.Series(probs, index = x_test.index)], axis=0)

            
        score = balanced_accuracy_score(dust_event_info.loc[predictions.index].sde_event, predictions)
        results.loc[balance, reg] = score



# Calculate the cross-validated metrics
input_dim = data.shape[1]
balance_mult = 2.5
reg = 1e-6
learn_rate = 1.5e-4
epoch_num = 25
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
 
    # batch size of 7 days
    train_dataloader = DataLoader(trainDataset, batch_size = 7 * 24, shuffle = False)
    
    input_dim = x_train.shape[1]

    # Model Initialization
    model = unet(input_dim)

    # Using the Weighted Loss between reconstruction loss and classification loss
    if balance_mult > 0:
        class_weight = torch.FloatTensor([balance_mult * (1-Y_train.mean())/Y_train.mean()])
    else:
        class_weight = torch.FloatTensor([1])

 
    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate, weight_decay= reg)
    

    print(f"Start fitting model {i}")  
    fit_model(model, epochs = epoch_num, trainLoader = train_dataloader, pos_weight=class_weight)
    print(f"Done fitting model {i}")
    
    
    test_tensor = torch.from_numpy(test_scale.to_numpy()).to(torch.float32)
    
    # Evaluate Model
    model.eval()
    f = nn.Sigmoid()
    probs = f(model(test_tensor)[0]).detach().numpy().reshape(-1)

    pred = (f(model(test_tensor)[0]).detach().numpy() >= 0.5).astype(int).reshape(-1)
    
    
    print(f"Balanced Accuracy of model {i} is {balanced_accuracy_score(Y_test, pred)}")
    
    # Concatenate the predictions for this split to the entire prediction series,
    # this allows the performance of the model to be judged for the entire year.
    predictions = pd.concat([predictions, pd.Series(pred, index = x_test.index)], axis = 0)
    predicted_probs = pd.concat([predicted_probs, pd.Series(probs, index = x_test.index)], axis=0)

   
    # Evaluate the concatenated predictions for all of 2020
evaluate_model(predictions, dust_event_info)
plot_predictions(pd.DataFrame({"UNET": predictions}, index = data.index), dust_event_info)



""" Fit the model on all the 2020 data, for use in 2017 """
np.random.seed(1)

reg_2017 = 1e-6
learn_rate_2017 = 1.5e-4
epoch_num_2017 = 25

x_train = data.drop("block_nr", axis = 1)
Y_train = Y[x_train.index]

x_test = data2017
Y_test = Y_2017[x_test.index]

sc = StandardScaler()
train_scale_2017 = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
test_scale_2017 = pd.DataFrame(sc.transform(x_test), columns = x_test.columns, index = x_test.index)
trainDataset_2017 = CustomDataset(train_scale_2017, Y_train)
testDataset_2017 = CustomDataset(test_scale_2017, Y_test)

# batch size of 7 days
train_dataloader_2017 = DataLoader(trainDataset_2017, batch_size = 7 * 24, shuffle = False)

input_dim = x_train.shape[1]

# Model Initialization
model = unet(input_dim)

# Using the Weighted Loss between reconstruction loss and classification loss
class_weight = torch.FloatTensor([2])

# Using an Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate_2017, weight_decay= reg_2017)


print("Start fitting model")  
fit_model(model, epochs = epoch_num_2017, trainLoader = train_dataloader_2017, pos_weight=class_weight)
print("Done fitting model")


test_tensor_2017 = torch.from_numpy(test_scale_2017.to_numpy()).to(torch.float32)

# Evaluate Model on 2017
model.eval()
f = nn.Sigmoid()
probs = f(model(test_tensor_2017)[0]).detach().numpy().reshape(-1)

pred = (f(model(test_tensor_2017)[0]).detach().numpy() >= 0.5).astype(int).reshape(-1)

predictions_2017 = pd.Series(pred, index = data2017.index, name = "U-NET")

evaluate_model(predictions_2017, dust_event_info_2017)
plot_predictions(predictions_2017, dust_event_info_2017, 
                 start = "2017-01-01 00:00:00")

# model_results_2020["U-NET"] = predictions
# model_results_2017["U-NET"] = predictions_2017
