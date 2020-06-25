import torch
import torch.nn as nn

class Embedding(nn.Module):
    """Embedding Module"""
    
    def __init__(self, embedding_dim, vocab_size, init_weight=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    
    def forward(self, x):
        """
            x: (batch_size, timesteps)
        """
        return self.embedding(x)

class Encoder(nn.Module):
    """Encoder Module"""
    def __init__(self, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder_network = nn.GRU(embedding_dim, hidden_dim, 2, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        """
            x: (batch_size, timesteps, embedding_dim)
        """
        return self.encoder_network(x)

class Decoder(nn.Module):
    """Decoder Module"""
    def __init__(self, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.decoder_network = nn.GRU(embedding_dim, hidden_dim, 2, batch_first=True, bidirectional=True)

    def forward(self, x, hidden):
        """
            x: (batch_size, timesteps, embedding_dim)
            hidden: (hidden_state, cell_state)
        """
        return self.decoder_network(x, hidden)

class Projection(nn.Module):
    """Projection Module"""
    def __init__(self, hidden_dim, vocab_size):
        super(Projection, self).__init__()
        self.projection_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
            x: (batch_size, timesteps, hidden_dim)
        """
        x = self.projection_layer(x)
        return nn.functional.softmax(x, dim=-1)

class Seq2Seq(nn.Module):
    """ Simple Seq2Seq model """

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.embedding_encoder = Embedding(embedding_dim, vocab_size)
        self.embedding_decoder = Embedding(embedding_dim, vocab_size)
        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.project = Projection(hidden_dim * 2, vocab_size)

    def forward(self, x, xlens, y):
        """
            x: (batch_size, timesteps)
        """
        batch_size, num_timesteps = y.size()
        embedding = self.embedding_encoder(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, xlens, batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(embedding)
        embedding = self.embedding_decoder(y)

        outputs = []

        for i in range(num_timesteps):
            ip = embedding[:, i, :].unsqueeze(1)
            output, hidden = self.decoder(ip, hidden)
            output = self.project(output)
            outputs.append(output)

        return torch.cat(outputs, dim=1)