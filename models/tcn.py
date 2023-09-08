"""
Title: tcn.py

Description:
    Trains and evaluates temporal convolutional network (TCN) models. Performs 
    hyperparameter tuning. It estimates 2020 performance over 6 folds. It then 
    trains a full model on all 2020 data for evaluation of 2017.

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
from utils.model_evaluation import evaluate_model
from utils.plotting import plot_predictions
from utils.calibration import plot_calibration_curve

# Define kernel which penalizes the model switching between states too much in
# its final predictions (has the effect of smoothing predictions)
transition_penalty_kernel = (torch.tensor([[-1, 1]], dtype=torch.float32)).view(1,1,-1)


class TCN(torch.nn.Module):
    def __init__(self, input_dim, conv_dim, kernel_size = 3):
        super().__init__()
        self.input_dim = input_dim
        self.conv_dim = conv_dim
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
        self.fully_connected = nn.Sequential(
            nn.Linear(self.conv_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
    
        
        self.transition_penalty_conv = nn.Conv1d(
            in_channels=1,  # Number of classes in your output
            out_channels=1,           # Single channel output
            kernel_size=2,            # Convolutional kernel size (captures pairs)
            stride=1,
            padding=0,
            bias=False,
        )
        
        self.transition_penalty_conv.weight = nn.Parameter(transition_penalty_kernel, requires_grad=False)
        self.make_probs = nn.Sigmoid()
        
 
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
        
        logits = self.fully_connected(x)
        
        
        # This penalizes probability transitions, not transitions in binary classification
        # I am not sure how to get a non-zero gradient using the probabilities so will try this first
        transition_weights = torch.abs(self.transition_penalty_conv(self.make_probs(logits).squeeze().unsqueeze(0).unsqueeze(0)))
        
        return torch.FloatTensor(logits), torch.FloatTensor(transition_weights.squeeze())
        



class WeightedLoss(nn.Module):
    def __init__(self, class_weight, transition_weight = 1):
        super().__init__()
        self.class_weight = class_weight
        self.trans_weight = transition_weight

    def forward(self, pred, labels, transitions):
        ce = nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
        classification_loss = ce(pred, labels)
        transition_loss = torch.norm(transitions)
        
        loss = classification_loss + (self.trans_weight * transition_loss)
        return loss, classification_loss.item(), transition_loss.item()
    

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
    

def fit_model(model, epochs, trainLoader, trans_weight):
    np.random.seed(1)
    losses = []
    class_losses = []
    trans_losses = []
    for epoch in range(epochs):
        for X, y in trainLoader:
          loss_batch = 0
          trans_batch = 0
          class_batch = 0
          # Output of Autoencoder
          y = y.unsqueeze(1)
          y_pred, transitions = model(X)
           

          # Calculating the loss function
          loss, class_loss, trans_loss, = loss_function(y_pred, y, transitions)
           
          # The gradients are set to zero,
          # the gradient is computed and stored.
          # .step() performs parameter update
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          loss_batch += loss.item()
          class_batch += class_loss
          trans_batch += trans_loss
          
          
        # Storing the losses in a list for plotting
        if epoch % 10 == 0:
            print("Epoch {:03d}: Loss: {:.3f}  Transition. Loss: {:.3f}  Class. Loss: {:.3f}".format(epoch, loss_batch, trans_batch, class_batch))
        losses.append(loss_batch)
        class_losses.append(class_batch)
        trans_losses.append(trans_batch)

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title("Weighted Loss", fontsize=18)
    
    plt.subplot(2, 2, 3)
    plt.plot(trans_losses)
    plt.title(f"Transition ({trans_weight})", fontsize=12)
    
    plt.subplot(2, 2, 4)
    plt.plot(class_losses)
    plt.title("Classification", fontsize=12)
    plt.tight_layout()
    plt.show()
    
  

"""Load the data, labels, and splits"""
data2020 = read_JFJ_data(
    r"\processed\data_imputed_2020_splits.csv"
    )
data2020["neg_AE_SSA"] = (data2020.AE_SSA < 0).astype(int)

data2017 = read_JFJ_data(r"\final\cleaned_impute.csv", date_range = ["2017-01-01 00:00:00","2017-12-31 23:59:59"])
data2017["neg_AE_SSA"] = (data2017["AE_SSA"] < 0).astype(int)

model_results_2020 = pd.read_csv("predictions_2020", index_col=0)
model_results_2017 = pd.read_csv("predictions_2017", index_col=0)


dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
dust_event_info_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")
Y_2017 = dust_event_info_2017.sde_event
Y = dust_event_info.sde_event

splits = pd.read_json("six_train_test_splits.json")
plotting_splits = pd.read_json("plotting_splits.json")



""" Hyperparameter tuning """
transition_weights = [0, 0.001, 0.01, 0.1]
regularization = [1e-6, 1e-5, 1e-4, 1e-3]
balance_mults = [0, 1, 2]

learn_rate = 1e-4
epoch_num = 20
conv_dim = 256
predictions = pd.Series(dtype=int)
for trans in transition_weights:
    results = pd.DataFrame(columns = regularization, index = balance_mults)
    for regs in regularization:
        for bal in balance_mults:
            for i, row in plotting_splits.iterrows():
                np.random.seed(1)
                
                x_train = data2020[data2020.block_nr.isin(row["trainSet"])].drop("block_nr", axis = 1)
                x_test = data2020[data2020.block_nr.isin(row["testSet"])].drop("block_nr", axis = 1)

                Y_train = Y[x_train.index]
                Y_test = Y[x_test.index]
                
                sc = StandardScaler()
                train_scale = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
                test_scale = pd.DataFrame(sc.transform(x_test), columns = x_test.columns, index = x_test.index)
                trainDataset = CustomDataset(train_scale, Y_train)
                testDataset = CustomDataset(test_scale, Y_test)
             
                # batch size of 256 hours ~ 10 days
                train_dataloader = DataLoader(trainDataset, batch_size = 256, shuffle = False)
                
                input_dim = x_train.shape[1]

                # Model Initialization
                model = TCN(input_dim = input_dim, conv_dim = conv_dim)

                # Using the Weighted Loss between reconstruction loss and classification loss
                if bal > 0:
                    class_weight = torch.FloatTensor([bal * (1-Y_train.mean())/Y_train.mean()])
                else:
                    class_weight = torch.FloatTensor([1])

                loss_function = WeightedLoss(class_weight=class_weight, transition_weight=trans)
             
                # Using an Adam Optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate, weight_decay=regs)
                

                print(f"Start fitting model {i}")  
                fit_model(model, epochs = epoch_num, trainLoader = train_dataloader, trans_weight=trans)
                print(f"Done fitting model {i}")
                
                
                test_tensor = torch.from_numpy(test_scale.to_numpy()).to(torch.float32)
                
                # Evaluate Model
                model.eval()
                f = nn.Sigmoid()
                probs = f(model(test_tensor)[0]).detach().numpy().reshape(-1)
                pred = (probs >= 0.5).astype(int)
                
                
                print(f"Balanced Accuracy of model {i} is {balanced_accuracy_score(Y_test, pred)}")
                
                predictions = pd.concat([predictions, pd.Series(pred, index = x_test.index)], axis = 0)
                
            score = balanced_accuracy_score(dust_event_info.loc[predictions.index].sde_event, predictions)
            results.loc[bal, regs] = score
    
    results.to_csv(f"tcn_cv_trans_{trans}.csv")



# Train the final model on 2020 data
transition_weight = 0.01
regularization = 1e-3
balance_mult = 0
learn_rate = 1e-4
epoch_num = 25
conv_dim = 256

predictions = pd.Series(dtype=int)
for i, row in plotting_splits.iterrows():
    np.random.seed(1)
    
    x_train = data2020[data2020.block_nr.isin(row["trainSet"])].drop("block_nr", axis = 1)
    x_test = data2020[data2020.block_nr.isin(row["testSet"])].drop("block_nr", axis = 1)

    Y_train = Y[x_train.index]
    Y_test = Y[x_test.index]
    
    sc = StandardScaler()
    train_scale = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
    test_scale = pd.DataFrame(sc.transform(x_test), columns = x_test.columns, index = x_test.index)
    trainDataset = CustomDataset(train_scale, Y_train)
    testDataset = CustomDataset(test_scale, Y_test)
 
    # batch size of 256 hours ~ 10 days
    train_dataloader = DataLoader(trainDataset, batch_size = 256, shuffle = False)
    
    input_dim = x_train.shape[1]

    # Model Initialization
    model = TCN(input_dim = input_dim, conv_dim = conv_dim)

    # Using the Weighted Loss between reconstruction loss and classification loss
    class_weight = torch.FloatTensor([1])
    loss_function = WeightedLoss(class_weight=class_weight, transition_weight=transition_weight)
 
    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-3)
    

    print(f"Start fitting model {i}")  
    fit_model(model, epochs = epoch_num, trainLoader = train_dataloader, trans_weight=transition_weight)
    print(f"Done fitting model {i}")
    
    
    test_tensor = torch.from_numpy(test_scale.to_numpy()).to(torch.float32)
    
    # Evaluate Model
    model.eval()
    f = nn.Sigmoid()
    probs = f(model(test_tensor)[0]).detach().numpy().reshape(-1)
    pred = (probs >= 0.5).astype(int)
    
    
    print(f"Balanced Accuracy of model {i} is {balanced_accuracy_score(Y_test, pred)}")
    
    predictions = pd.concat([predictions, pd.Series(pred, index = x_test.index)], axis = 0)

evaluate_model(predictions, dust_event_info)
plot_predictions(pd.Series(predictions, name = "TCN"), dust_event_info)



# Train final model on 2020 data
np.random.seed(1)

x_train = data2020.drop("block_nr", axis = 1)
x_test = data2017

Y_train = Y[x_train.index]
Y_test = Y_2017[x_test.index]

sc = StandardScaler()
train_scale = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
test_scale = pd.DataFrame(sc.transform(x_test), columns = x_test.columns, index = x_test.index)
trainDataset = CustomDataset(train_scale, Y_train)
testDataset = CustomDataset(test_scale, Y_test)

# batch size of 256 hours ~ 10 days
train_dataloader = DataLoader(trainDataset, batch_size = 256, shuffle = False)

input_dim = x_train.shape[1]

# Model Initialization
model = TCN(input_dim = input_dim, conv_dim = conv_dim)

# Using the Weighted Loss between reconstruction loss and classification loss
class_weight = torch.FloatTensor([1])
loss_function = WeightedLoss(class_weight=class_weight, transition_weight=transition_weight)

# Using an Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-3)


print("Start fitting model")  
fit_model(model, epochs = epoch_num, trainLoader = train_dataloader, trans_weight=transition_weight)
print("Done fitting model")


# torch.save(model.state_dict(), "tcn.pth")

test_tensor = torch.from_numpy(test_scale.to_numpy()).to(torch.float32)

# Evaluate Model
model.eval()
f = nn.Sigmoid()
probs = f(model(test_tensor)[0]).detach().numpy().reshape(-1)
pred = (probs >= 0.5).astype(int)


print(f"Balanced Accuracy of model is {balanced_accuracy_score(Y_test, pred)}")

predictions_2017 = pd.Series(pred, index = data2017.index, name = "TCN")
evaluate_model(predictions_2017, dust_event_info_2017)
plot_predictions(predictions_2017, dust_event_info_2017, start = "2017-01-01 00:00:00")
