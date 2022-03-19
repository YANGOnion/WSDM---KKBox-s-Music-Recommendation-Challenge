
from pathlib import Path
import pyarrow.feather as feather
import torch
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score

from data_utils import *
from model import *


DATA_PATH = 'D:\MLData\WSDM-KKbox'

RARE_THRES = 100
VILID_SIZE = 0.2
BATCH_SIZE = 1024
DIM_EMBED = 5
DIM_DEEP = [20, 20, 20]
LEARNING_RATE = 1e-4
N_EPOCHS = 100
PATIENCE = 10

### create dataset
df = feather.read_feather(Path(DATA_PATH)/'process/preproc_test.feather')
df = create_feature(df)
n_fields, n_features, X_train_index, X_train_value, y_train, X_valid_index, X_valid_value, y_valid, \
    X_test_index, X_test_value = data_preprocess(df, RARE_THRES, VILID_SIZE, 'target')
loader = Data.DataLoader(
    Data.TensorDataset(X_train_index, X_train_value, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)
loader_valid = Data.DataLoader(
    Data.TensorDataset(X_valid_index, X_valid_value, y_valid),
    batch_size=65536,
    shuffle=False
)
loader_test = Data.DataLoader(
    Data.TensorDataset(X_test_index, X_test_value),
    batch_size=65536,
    shuffle=False
)


### training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepFM(n_fields, n_features, DIM_EMBED, DIM_DEEP)
model.to(device)
print('# parameters = ', sum(p.numel() for p in model.parameters()))
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_train = []
loss_valid = []
auc_train = []
auc_valid = []
max_auc = 0
for epoch in range(N_EPOCHS):
    
    # training
    pred_train = []
    obs_train = []
    for X1, X2, y in loader:
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X1, X2)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred_train.append(y_pred.cpu())
            obs_train.append(y.cpu())
        del X1, X2, y, y_pred
        torch.cuda.empty_cache()
    pred_train = torch.cat(pred_train)
    obs_train = torch.cat(obs_train)
    loss_train.append(loss_fn(pred_train, obs_train).item())
    auc_train.append(roc_auc_score(obs_train.numpy(), pred_train.numpy()))
    
    # validating
    with torch.no_grad():
        pred_valid = []
        for X1, X2, y in loader_valid:
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            y_pred = model(X1, X2)
            pred_valid = pred_valid + [y_pred.cpu()]
            del X1, X2, y, y_pred
            torch.cuda.empty_cache()
        pred_valid = torch.cat(pred_valid)
        loss_valid.append(loss_fn(pred_valid, y_valid).item())
        auc_valid.append(roc_auc_score(y_valid.numpy(), pred_valid.numpy()))
    
    print(epoch, ":", loss_train[-1], loss_valid[-1], ";", auc_train[-1], auc_valid[-1])
    
    if auc_valid[-1]>=max_auc:
        torch.save(model, Path(DATA_PATH)/'model/deepfm.pth')
        max_auc = auc_valid[-1]
    elif max(auc_valid[-min(len(auc_valid), PATIENCE):])<max_auc:
        break
        

### testing
final_model = torch.load(Path(DATA_PATH)/'model/deepfm.pth')
final_model.to(device)
with torch.no_grad():
    pred_test = []
    for X1, X2 in loader_test:
        X1, X2 = X1.to(device), X2.to(device)
        y_pred = final_model(X1, X2)
        pred_test = pred_test + [y_pred.cpu()]
        del X1, X2, y_pred
        torch.cuda.empty_cache()
    pred_test = np.concatenate(pred_test).ravel()
pd.DataFrame({'id': range(len(pred_test)), 'target': pred_test}).to_csv(Path(DATA_PATH)/'model/submit_deepfm_lr-4.csv', index = False)


### tracking
import seaborn as sns
df_epoch = pd.DataFrame({'epoch': range(len(loss_train)), 'loss_train': loss_train, 'loss_valid': loss_valid})
sns.relplot(data=df_epoch.melt(id_vars = 'epoch'), x='epoch', y='value', hue='variable', kind='line')

df_epoch = pd.DataFrame({'epoch': range(len(auc_train)), 'auc_train': auc_train, 'auc_valid': auc_valid})
sns.relplot(data=df_epoch.melt(id_vars = 'epoch'), x='epoch', y='value', hue='variable', kind='line')