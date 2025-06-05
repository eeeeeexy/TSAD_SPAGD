from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class nconv_gat(nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, heads=1, dropout=0.3):
        super(nconv_gat, self).__init__()
        self.edge_index = edge_index  # [2, E]
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)

    def forward(self, x):
        # x: [B, C, T, N]
        B, C, T, N = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, N, C]
        x = x.view(B * T, N, C)  # [B*T, N, C]

        out = []
        for xt in x:  # xt: [N, C]
            out.append(self.gat(xt, self.edge_index))  # [N, out_channels]

        out = torch.stack(out, dim=0)  # [B*T, N, out_channels]
        out = out.view(B, T, N, -1).permute(0, 3, 1, 2).contiguous()  # [B, out_channels, T, N]
        return out

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        # adj normalization
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        # graph propagation
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h,a)  # alpha=0.05
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho

from torch_geometric.nn import GATConv
class mixprop_gat_init(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha, heads=1):
        super(mixprop_gat_init, self).__init__()
        self.gdep = gdep
        self.alpha = alpha

        self.gat_layers = nn.ModuleList([
            GATConv(c_in, c_in, heads=heads, dropout=dropout, concat=False) for _ in range(gdep)
        ])

        self.mlp = nn.Linear((gdep + 1) * c_in, c_out)

    def forward(self, x, edge_index):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.gat_layers[i](h, edge_index)
            out.append(h)
        
        ho = torch.cat(out, dim=-1)
        return self.mlp(ho)
    
class mixprop_gat(nn.Module):
    def __init__(self, c_in, c_out, gdep, edge_index, dropout=0.3, alpha=0.05, heads=1):
        super(mixprop_gat, self).__init__()
        self.gconvs = nn.ModuleList([
            nconv_gat(c_in, c_in, edge_index, heads=heads, dropout=dropout) for _ in range(gdep)
        ])
        self.mlp = nn.Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):
        # x: [B, C, T, N]
        h = x
        out = [h]

        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.gconvs[i](h)
            out.append(h)

        ho = torch.cat(out, dim=1)  # [B, (gdep+1)*C, T, N]
        ho = ho.permute(0, 2, 3, 1)  # [B, T, N, C']
        ho = self.mlp(ho)            # [B, T, N, c_out]
        ho = ho.permute(0, 3, 1, 2)  # [B, c_out, T, N]
        return ho
    

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2
    

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = 94
        self.out_features = 16

        self.W = nn.Linear(self.in_features, self.out_features, bias=False)
        self.attn = nn.Parameter(torch.empty(size=(2 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)

        self.linear = nn.Linear(65536, 32)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)

        # Unsqueeze to prepare for repetition
        Wh_unsqueezed_1 = Wh.unsqueeze(2)  # Shape: [16, 16, 1, 51, 16]
        Wh_unsqueezed_2 = Wh.unsqueeze(1)  # Shape: [16, 1, 16, 51, 16]

        # Repeat along the appropriate dimensions:
        Wh_repeat_1 = Wh_unsqueezed_1.repeat(1, 1, N, 1, 1)  # Repeat across the 3rd axis (dim=2)
        Wh_repeat_2 = Wh_unsqueezed_2.repeat(1, N, 1, 1, 1)  # Repeat across the 2nd axis (dim=1)

        # Concatenate them along the last dimension
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=4)  

        a_input_transformed = self.linear(a_input.view(-1, 65536))  # Flatten and apply linear transformation
        a_input_transformed = a_input_transformed.view(16, 16, 32)
        # a_input = a_input.view(N, N, -1)

        e = self.leakyrelu(torch.matmul(a_input_transformed, self.attn).squeeze(2))  # (N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)


class mixatt(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixatt, self).__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.gat_layers = nn.ModuleList([GATLayer(c_in, c_in, dropout, alpha) for _ in range(gdep)])
        self.mlp = linear((gdep + 1) * c_in, c_out)

    def forward(self, x, adj):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.gat_layers[i](h, adj)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        ho = self.dropout(ho)
        return ho


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            # return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
