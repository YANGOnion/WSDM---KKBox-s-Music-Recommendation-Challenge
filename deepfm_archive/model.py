
import torch.nn as nn

class DeepFM(nn.Module):
    
    def __init__(self, n_fields, n_features, dim_embed, 
                 dim_deep = [100, 60, 20]):
        super(DeepFM, self).__init__()
        self.layer_embed = nn.Embedding(n_features, dim_embed)
        self.layer_w = nn.Embedding(n_features, 1)
        self.layer_linear = nn.Linear(1, 1)
        dim_deep = [n_fields*dim_embed] + dim_deep
        seq_modules = []
        for i in range(len(dim_deep)-1):
            seq_modules.append(nn.Linear(dim_deep[i], dim_deep[i+1]))
            seq_modules.append(nn.BatchNorm1d(dim_deep[i+1])),
            seq_modules.append(nn.ReLU())
            seq_modules.append(nn.Dropout(0.5))
        seq_modules.append(nn.Linear(dim_deep[-1], 1))
        self.layer_deep = nn.Sequential(*seq_modules)
        self.layer_act = nn.Sigmoid()
        
    def forward(self, X1, X2):
        embed = self.layer_embed(X1)
        embed = embed * X2.view(X2.shape[0], X2.shape[1], 1).repeat(1, 1, self.layer_embed.embedding_dim)
        embed_fm = embed.sum(axis=1)**2 - (embed**2).sum(axis=1)
        embed_fm = 0.5 * embed_fm.sum(axis=1).reshape(-1, 1)
        w = (self.layer_w(X1)*X2.view(X2.shape[0], X2.shape[1], 1)).sum(axis=1)
        embed_fm = self.layer_linear(w + embed_fm)
        concat_deep = embed.reshape(X1.shape[0], -1)
        concat_deep = self.layer_deep(concat_deep)
        return self.layer_act(embed_fm + concat_deep)