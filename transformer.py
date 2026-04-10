import math
import torch
import torch.utils.data


def get_positional_embeddings(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1).float()
    two_i = torch.arange(d_model // 2) * 2
    emb = torch.empty(seq_len, d_model)
    emb[:, 0::2] = torch.sin(pos / 10000 ** (two_i / d_model))
    emb[:, 1::2] = torch.cos(pos / 10000 ** (two_i / d_model))
    return emb


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        dk_dim = k.shape[-1]
        qk = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        x = qk / math.sqrt(dk_dim)
        if mask is not None:
            x = x.masked_fill(~mask.bool(), float('-inf'))
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v.permute(0, 2, 1, 3))
        return x



class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_heads = n_head # 8 (from the paper)
        self.d_model = d_model
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.scaled = ScaledDotProductAttention()
        self.final_linear = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch = k.shape[0]
        sequence = q.shape[1] # output length is always q length
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        # input should be dk, so we split everything
        k = torch.reshape(k, (k.shape[0], k.shape[1], self.n_heads, self.d_model // self.n_heads))
        q = torch.reshape(q, (q.shape[0], q.shape[1], self.n_heads, self.d_model // self.n_heads))
        v = torch.reshape(v, (v.shape[0], v.shape[1], self.n_heads, self.d_model // self.n_heads))
        scaled = self.scaled(q, k, v, mask)
        scaled = scaled.permute(0, 2, 1, 3).reshape((batch, sequence, self.d_model))
        result = self.final_linear(scaled)
        return result

class EncoderLayer(torch.nn.Module): # was TransformerLayer
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.multihead_self_attention = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model * 4, d_model))
        self.layer_norm_attention = torch.nn.LayerNorm(d_model)
        self.layer_norm_feed = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm_attention(x + self.dropout(self.multihead_self_attention(x, x, x)))
        x = self.layer_norm_feed(x + self.dropout(self.feed_forward(x)))
        return x



class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.multihead_self_attention_enc = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.multihead_self_attention = MultiHeadSelfAttention(d_model, n_head, dropout)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model * 4, d_model))
        self.layer_norm_attention_enc = torch.nn.LayerNorm(d_model)
        self.layer_norm_attention = torch.nn.LayerNorm(d_model)
        self.layer_norm_feed = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        # output dimension should be 512
        # always separate LayerNorm for different layers

    def forward(self, x, enc_out, mask):
        x = self.layer_norm_attention_enc(x + self.dropout(self.multihead_self_attention_enc(x, x, x, mask=mask)))
        x = self.layer_norm_attention(x + self.dropout(self.multihead_self_attention(x, enc_out, enc_out)))
        x = self.layer_norm_feed(x + self.dropout(self.feed_forward(x)))
        return x



class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.input_projection = torch.nn.Linear(self.input_dim, self.d_model)
        self.transformer_layers = torch.nn.ModuleList([EncoderLayer(self.d_model, self.n_head, dropout) for i in range(self.n_layers)])
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_projection(x)
        pos = get_positional_embeddings(x.shape[1], self.d_model).to(x.device)
        x = self.dropout(x + pos)
        for i in range(len(self.transformer_layers)):
            x = self.transformer_layers[i](x)
        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.input_projection = torch.nn.Linear(self.input_dim, self.d_model)
        self.transformer_layers = torch.nn.ModuleList([DecoderLayer(self.d_model, self.n_head, dropout) for i in range(self.n_layers)])
        self.final_layer = torch.nn.Linear(self.d_model, input_dim) # to spectrogram dim
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, targets, enc_out):
        targets = self.input_projection(targets)
        seq_len = targets.shape[1]
        pos = get_positional_embeddings(targets.shape[1], self.d_model).to(targets.device)
        x = self.dropout(targets + pos)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(targets.device)
        for i in range(len(self.transformer_layers)):
            x = self.transformer_layers[i](x, enc_out, mask=mask)
        x = self.final_layer(x)

        return x


class Transformer(torch.nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.encoder = TransformerEncoder(self.input_dim, self.d_model, self.n_head, self.n_layers, dropout)
        self.decoder = TransformerDecoder(self.input_dim, self.d_model, self.n_head, self.n_layers, dropout)

    def forward(self, x, targets):
        enc_out = self.encoder(x)
        x = self.decoder(targets, enc_out)
        return x


# Fake dataset
batch_size = 32
seq_len = 32
input_dim = 80
n_batches = 20

inputs  = torch.randn(n_batches * batch_size, seq_len, input_dim)
targets = inputs.clone() # copying task
dataset = torch.utils.data.TensorDataset(inputs, targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
model = Transformer(input_dim=input_dim, d_model=512, n_head=8, n_layers=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

n_epochs = 20


model.train()

for epoch in range(n_epochs):
    total_loss = 0.0

    for x, targets in dataloader:
        x = x.to(device)
        targets = targets.to(device)
        dec_input = torch.zeros_like(targets)

        optimizer.zero_grad()
        output = model(x, dec_input)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{n_epochs}  loss: {total_loss / len(dataloader):.4f}")
