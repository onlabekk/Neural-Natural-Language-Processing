class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.nn.Parameter(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)), requires_grad=False)
        
    def forward(self, query, key, value, mask=None):
        
        bs = query.shape[0]
        
        q, k, v = self.fc_q(query), self.fc_k(key), self.fc_v(value) 
                
        seq_len = query.shape[1]
        q, k, v = q.view(bs, -1, self.n_heads, self.head_dim).transpose(1,2), \
                  k.view(bs, -1, self.n_heads, self.head_dim).transpose(1,2), \
                  v.view(bs, -1, self.n_heads, self.head_dim).transpose(1,2)
        
        energy = q @ k.transpose(-2, -1) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill_(mask.unsqueeze(1).unsqueeze(1), 1e-9)
        
        attention = torch.softmax(energy, dim=-1)
        
        x = self.dropout(attention) @ v
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bs, -1, self.hid_dim)
        
        x = self.fc_o(x)
        
        return x