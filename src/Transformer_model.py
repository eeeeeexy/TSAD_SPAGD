#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TriangularCausalMask():

    def __init__(self, L, device="cpu"):
 
        # Define the shape of the mask tensor
        mask_shape = [L, L]
        
        with torch.no_grad():

            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            
    @property
    def mask(self):
        """
        Property to get the mask tensor.
        
        Returns:
        torch.Tensor: The mask tensor.
        """
        return self._mask
    


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        
        # Padding depending on PyTorch version
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        
        # 1D convolutional layer to project each token of the input to a dense vector
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        
        # Initialization for the convolution layer for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Convert tokens to dense vectors using the convolutional layer
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        
        # Compute the positional encodings once in log space.
        # This encoding will remain fixed across all batches and epochs.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 'position' corresponds to the position in the sequence.
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        # 'div_term' scales down the positional encoding, especially for larger dimensions.
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Generate sinusoidal positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store the positional encodings in the buffer so it's not treated as a model parameter
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Return positional encoding for each position of the input 'x'
        return self.pe[:, :x.size(1)]



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        # Token embedding layer
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # Positional embedding layer
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Combine token and positional embeddings and apply dropout
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        
        # Generate the mask for the current batch size and sequence length
        attn_mask = TriangularCausalMask(L=x.size(1), device=x.device).mask

        attn_output, _ = self.attention(x, x, x, attn_mask=None)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)
    
    

class Encoder(nn.Module):
    def __init__(self, encoder_layers, rbf_layer=None, modified_d_model=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.rbf_layer = rbf_layer
        self.modified_d_model = modified_d_model
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        rbf_out = None
        second_layer_out = None
        
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            

            if idx == 1:  # After processing through the second layer
                second_layer_out = x  # Save the output of the second layer
                if self.rbf_layer:
                    rbf_out = self.rbf_layer(x)
                    x = rbf_out
                    
        if self.norm:
            x = self.norm(x)
        return x, rbf_out, second_layer_out



class Reconstrcution_model(nn.Module):
    def __init__(self, input_size, d_model, n_heads, e_layers, d_ff, dropout, activation='gelu'):
        super(Reconstrcution_model, self).__init__()
        self.e_layers = e_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.input_size = input_size


        # Encoding
        self.embedding = DataEmbedding(c_in=input_size, d_model=32, dropout=0.0)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff, dropout=self.dropout, activation="relu")
                for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        
        self.projection = nn.Linear(self.d_model, self.input_size)

    def forward(self, x):
        # x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = self.embedding(x)
        enc_out, _,  second_layer_out = self.encoder(x)

        dec_out = self.projection(enc_out)
        
        return dec_out, second_layer_out, None
    
  