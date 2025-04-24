# models/full_model.py
import torch.nn as nn
from models.semantic_encoder import SemanticEncoder
from models.cnn_encoder import CNNEncoder
from models.channel_layer import ChannelLayer
from models.cnn_decoder import CNNDecoder
from models.semantic_decoder import SemanticDecoder

class SemanticCommSystem(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = SemanticEncoder(vocab_size)
        self.cnn_enc = CNNEncoder()
        self.channel = ChannelLayer(snr_db=10, rician_k=3.0)
        self.cnn_dec = CNNDecoder()
        self.decoder = SemanticDecoder(vocab_size)

    def forward(self, src_tokens, tgt_tokens):
        x = self.encoder(src_tokens)
        x = self.cnn_enc(x)
        x = self.channel(x)
        x = self.cnn_dec(x)
        logits = self.decoder(tgt_tokens, x)
        return logits
