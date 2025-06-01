import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import expit

import numpy as np
from torch import Tensor
import math, scipy

from src.Transformer_model import Reconstrcution_model
from src.layer_ import *
from torch_geometric.nn import GATConv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dot_product_decode(Z):

    Z_ = torch.matmul(Z, Z.transpose(1,0))
    mean = Z_.mean()
    std = Z_.std()
    Z_ = (Z_ - mean) / (std + 1e-8)
    Z_1 = Z_.detach()
    A_pred = torch.sigmoid(Z_1)
    return A_pred


def trans_adj_to_edge_index(adj):
    adj = scipy.sparse.coo_matrix(adj.cpu().detach().numpy())
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    adj = torch.LongTensor(indices)
    return adj


class graph_constructor(nn.Module):

    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None): # k: 20: dim: 256
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, inputs, inputs_init, outputs_init, idx):

        device = self.emb1.weight.device
        idx = idx.to(device)

        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        # node consin similarity
        nodevec = inputs.squeeze(1)
        nodevec = nodevec.transpose(1,0)
        nodevec = nodevec.reshape(inputs.shape[-2], nodevec.size(1)*nodevec.size(2))

        kkk = dot_product_decode(nodevec)

        gap_list = torch.mean(torch.mean(torch.abs(inputs_init-outputs_init), dim=1), dim=0)
        gap_list_detach = gap_list.detach()
        gap_list_ = torch.sigmoid(gap_list_detach)
        # compute k%
        k = 30  # 前5%
        num = math.ceil(len(gap_list_) * (k / 100))
        topk_, topk_idx = torch.topk(gap_list_, num, largest=False)
        topk_ = topk_.flip(0)
        topk_idx = topk_idx.flip(0)
        threshold = torch.mean(topk_)

        # add the reconstruction gap
        try:
            cand_ = [index for index, item in enumerate(topk_) if item > threshold]
            anomaly_list = topk_idx[:cand_[-1]+1]
            anomaly_value = topk_[:cand_[-1]+1]
            anomaly_value = anomaly_value.unsqueeze(1)
        except:
            import pdb; pdb.set_trace()

        # add value on the raw
        kkk[anomaly_list] += anomaly_value
        # add value on the colmn
        for id_, val in enumerate(anomaly_value):
            kkk[:, anomaly_list[id_]] += val
        # delete the duplicated value
        kkk[anomaly_list, anomaly_list] -= anomaly_value.squeeze(1)

        adj = kkk
        mask = torch.zeros(idx.size(0), idx.size(0)).to(device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask = mask.to(device)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask

        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class stgnn_gat_improve(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, devices=1, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=0.1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(stgnn_gat_improve, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1)).to(device)
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, devices, alpha=tanhalpha, static_feat=static_feat).to(device)

        self.input_proj = nn.Linear(1504, 16).to(device)


        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation).to(device))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation).to(device))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)).to(device))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)).to(device))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)).to(device))

                self.gconv1.append(GATConv(conv_channels, residual_channels, heads=1, dropout=dropout).to(device))
                self.gconv2.append(GATConv(conv_channels, residual_channels, heads=1, dropout=dropout).to(device))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline).to(device))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline).to(device))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True).to(device)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True).to(device)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True).to(device)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True).to(device)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True).to(device)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True).to(device)

        self.idx = torch.arange(self.num_nodes)

    def forward(self, input, input_init, output_init, idx=None):

        seq_len = input.size(3)

        try:
            assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        except:
            import pdb; pdb.set_trace()

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        input = input.to(device)
        input_init = input_init.to(device)
        output_init = output_init.to(device)
        self.idx = self.idx.to(device)

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(input, input_init, output_init, self.idx).to(device)
                else:
                    adp = self.gc(input, input_init, output_init, idx).to(device)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            edge_index = adp.nonzero().t()
            if self.gcn_true:

                # GAT
                B, C, N, T = x.shape
                x_out = []
                for t in range(T):
                    x_t = x[:, :, :, t]
                    x_t = x_t.permute(0, 2, 1)
                    x_t = x_t.reshape(B * N, C)

                    # edge_index: shape [2, num_edges] — shared across all batches
                    # Repeat for each batch
                    edge_index_batch = []
                    for b in range(B):
                        edge_index_batch.append(edge_index + b * N)
                    edge_index_all = torch.cat(edge_index_batch, dim=1)

                    # Apply GAT
                    x_t_out = self.gconv1[i](x_t, edge_index_all) + self.gconv2[i](x_t, edge_index_all)
                    x_t_out = x_t_out.view(B, N, -1).permute(0, 2, 1)
                    x_out.append(x_t_out.unsqueeze(-1))
                # Stack over time
                x = torch.cat(x_out, dim=-1)

            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        x = torch.transpose(x, 1, 3)

        return x, adp


class SPAGD_model(nn.Module):
    def __init__(self, dataset, device, args):
        super(SPAGD_model, self).__init__()

        self.device = device
        self.reconstruct = Reconstrcution_model(input_size=dataset.x.shape[1], d_model=32, n_heads=8, e_layers=3, d_ff=128, dropout=0.0).to(self.device)

        gcn_true = args.graph_model
        buildA_true = True
        gcn_depth = 2
        num_nodes = args.node_size
        predefined_A = None
        subgraph_size = args.node_size
        node_dim = 256
        dilation_exponential = 1
        dropout = 0.1
        conv_channels = 16
        residual_channels = 16
        skip_channels = 32
        end_channels = 32
        
        in_dim = 1
        
        layers = 2
        propalpha = 0.1
        tanhalpha = 20

        self.chunk_num = 5
        seq_in_len = args.window_size
        seq_in_len_t = int(args.window_size/self.chunk_num)
        seq_out_len = 20  

        self.stgnn_s_input = stgnn_gat_improve(gcn_true, buildA_true, gcn_depth, num_nodes, predefined_A=predefined_A,
                        dropout=dropout, subgraph_size=subgraph_size,
                        node_dim=node_dim,
                        dilation_exponential=dilation_exponential,
                        conv_channels=conv_channels, residual_channels=residual_channels,
                        skip_channels=skip_channels, end_channels=end_channels,
                        seq_length=seq_in_len, in_dim=in_dim, out_dim=seq_out_len,
                        layers=layers, propalpha=propalpha, tanhalpha=tanhalpha, layer_norm_affline=True)
        self.stgnn_t_input = stgnn_gat_improve(gcn_true, buildA_true, gcn_depth, num_nodes, predefined_A=predefined_A,
                    dropout=dropout, subgraph_size=subgraph_size,
                    node_dim=node_dim,
                    dilation_exponential=dilation_exponential,
                    conv_channels=conv_channels, residual_channels=residual_channels,
                    skip_channels=skip_channels, end_channels=end_channels,
                    seq_length=seq_in_len_t, in_dim=in_dim, out_dim=seq_out_len,
                    layers=layers, propalpha=propalpha, tanhalpha=tanhalpha, layer_norm_affline=True)
            
        self.graph_model = args.graph_model

        if args.dataset == 'smap':
            self.fc1 = nn.Linear(3000, 256).to(device)
        if args.dataset == 'psm':
            self.fc1 = nn.Linear(3120, 256).to(device)
        if args.dataset == 'smd':
            self.fc1 = nn.Linear(4560, 256).to(device)
        if args.dataset == 'msl':
            self.fc1 = nn.Linear(6600, 256).to(device)
        if args.dataset == 'msl_TranAD':
            self.fc1 = nn.Linear(6600, 256).to(device)
        if args.dataset == 'swat':
            self.fc1 = nn.Linear(6120, 256).to(device)
        if args.dataset == 'wadi':
            self.fc1 = nn.Linear(15240, 256).to(device)
        self.fc2 = nn.Linear(256, args.window_size).to(device)

    def forward(self, inputs: Tensor) -> Tensor:

        # Reconstruct
        inputs = inputs.to(device)
        outputs, _, _ = self.reconstruct(inputs)

        # construct the spatial graph
        inputs_s = torch.transpose(inputs, 1, 2)
        inputs_s = torch.unsqueeze(inputs_s, 1)
        outputs_s = torch.transpose(outputs, 1, 2)
        outputs_s = torch.unsqueeze(outputs_s, 1)

        inputs_s = inputs_s.to(device)
        outputs_s = outputs_s.to(device)

        # ==== spatial graph input and output : inputs_x & outputs_x
        inputs_x, inputs_adp = self.stgnn_s_input(inputs_s, inputs, outputs)
        outputs_x, outputs_adp = self.stgnn_s_input(outputs_s, inputs, outputs)

        input_chunks_ = torch.chunk(inputs, chunks=self.chunk_num, dim=1)
        output_chunks_ = torch.chunk(outputs, chunks=self.chunk_num, dim=1)

        # ==================== Input ====================#

        # ==== construct the temporal graph ==== # 
        input_chunks_ti_ = torch.chunk(inputs_s, chunks=self.chunk_num, dim=3)
        input_chunks_ti = [inputs_x]
        for idx_, chunk_ti in enumerate(input_chunks_ti_):
            input_ti, inputs_adp_ti = self.stgnn_t_input(chunk_ti, input_chunks_[idx_], output_chunks_[idx_])
            input_chunks_ti.append(input_ti)
        input_st = torch.concat(input_chunks_ti, dim=3)

        # ==== spatial and temporal embedding -> prediction (label: all 0)
        input_st = torch.squeeze(input_st, 1)
        input_feature = input_st.transpose(1, 2)
        input_st = torch.reshape(input_st, (input_st.size(0), input_st.size(1)*input_st.size(2)))
        input_st_fc1 = self.fc1(input_st)
        final_input_predict = self.fc2(input_st_fc1)

        inputs_label = torch.zeros(final_input_predict.size(0), final_input_predict.size(1)).to(device=0)

        # ===================== Output (Reconstructed) ========================= #
        
        # ==== construct the temporal graph
        output_chunks_ti_ = torch.chunk(outputs_s, chunks=self.chunk_num, dim=3)
        output_chunks_ti = [outputs_x]
        for idx_, chunk_ti in enumerate(output_chunks_ti_):
            output_ti, outputs_adp_ti = self.stgnn_t_input(chunk_ti, input_chunks_[idx_], output_chunks_[idx_])
            output_chunks_ti.append(output_ti)
        output_st = torch.concat(output_chunks_ti, dim=3)

        # ==== spatial and temporal embedding -> prediction (label: all 1)
        output_st = torch.squeeze(output_st, 1)
        output_feature = output_st.transpose(1, 2)
        output_st = torch.reshape(output_st, (output_st.size(0), output_st.size(1)*output_st.size(2)))
        output_st_fc1 = self.fc1(output_st)
        final_output_predict = self.fc2(output_st_fc1)
        outputs_label = torch.ones(final_output_predict.size(0), final_output_predict.size(1)).to(device=0)

        return outputs, final_input_predict, final_output_predict, inputs_label, outputs_label, input_feature, output_feature
