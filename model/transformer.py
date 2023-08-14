import torch 
import torch.nn as nn 


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        
        self.hidden_dim = hidden_dim 
        self.n_heads = n_heads 
        self.dropout_ratio = dropout_ratio 
        self.head_dim = hidden_dim // n_heads 
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim) 
        self.fc_v = nn.Linear(hidden_dim, hidden_dim) 
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio) 
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value) 
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale 
        
        attention = torch.softmax(energy, dim=-1) 
        
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        x = x.view(batch_size, -1, self.hidden_dim)
        
        x = self.fc_o(x)
        return x, attention 


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim) 
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x 


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim) 
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio) 
    
    def forward(self, x):
        _x, attention = self.self_attention(x, x, x)
        x = self.self_attn_layer_norm(x + self.dropout(_x))
        _x = self.positionwise_feedforward(x)
        x = self.ff_layer_norm(x + self.dropout(_x))
        return x, attention


class SpecTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_len=500):
        super().__init__()
        
        self.device = device
        self.pos_embedding = nn.Embedding(max_len, input_dim).to(device)
        self.hidden_embedding = nn.Linear(input_dim, hidden_dim)
        self.reconstruction = nn.Linear(hidden_dim, input_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim])).to(device)
        
    def forward(self, x):
        if len(x.shape) == 3:
            batch_size, freq, _ = x.shape 
        else:
            batch_size = 1
            freq, _ = x.shape 
        
        pos = torch.arange(0, freq).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        x = self.dropout(x * self.scale + self.pos_embedding(pos))
        x = self.hidden_embedding(x)
        
        for layer in self.layers:
            x, attention_map = layer(x)
        
        x = self.reconstruction(x)
        return x, attention_map