import torch
from torch import nn
import numpy as np
from .mlp import  MultiLayerPerceptron
import torch.nn.functional as F
import pickle
import math



class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                              padding_mode='replicate', bias=True)
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out

class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.q = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

        self.memory = nn.Parameter(torch.randn(head, seq_length, num_nodes, self.d_k))
        nn.init.xavier_uniform_(self.memory)

        self.weight = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        apt_size = 10
        nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        self.nodevec1, self.nodevec2 = [
            nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs
        ]
        self.weights_pool = nn.init.xavier_normal_(
            nn.Parameter(
                torch.FloatTensor(self.seq_length, 3, 3)
            )
        )
        self.bias_pool = nn.init.xavier_normal_(
            nn.Parameter(torch.FloatTensor(num_nodes, self.d_k))
        )
        self.attn_metrix = []

    def forward(self, input):
        query, value = self.q(input), self.v(input)
        query = query.view(
            query.shape[0], -1, self.d_k, query.shape[2], self.seq_length
        ).permute(0, 1, 4, 3, 2)
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], self.seq_length
        ).permute(0, 1, 4, 3, 2)

        key = torch.softmax(self.memory / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        Aapt = torch.softmax(
            F.relu(torch.matmul(self.nodevec1, self.nodevec2)), dim=-1
        )
        kv = torch.einsum("hlnx, bhlny->bhlxy", key, value)
        attn_qkv = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)

        weights = torch.einsum("nm,dio->nio", Aapt, self.weights_pool)  # N, cheb_k*in_dim, out_dim
        bias = torch.matmul(Aapt, self.bias_pool)
        attn_dyn = torch.einsum("bhlnc,nio->bhlnc", value, weights) + bias[None, None, None, :, :]

        x = attn_qkv + attn_dyn

        self.attn_metrix = x
        x = (
            x.permute(0, 1, 4, 3, 2)
                .contiguous()
                .view(x.shape[0], self.d_model, self.num_nodes, self.seq_length)
        )
        x = self.concat(x)
        if self.num_nodes not in [170, 358, 5]:
            x = x * self.weight + self.bias + x
        return x, self.weight, self.bias


class Encoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = SpatialAttention(
            device, d_model, head, num_nodes, seq_length=seq_length
        )
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input):
        # 64 64 170 12
        x, weight, bias = self.attention(input)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh)
        return xh


class MANet(nn.Module):


    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels= self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
                          int(self.if_spatial) + self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)

        # spacial att
        self.SpatialBlock = Encoder(
            self.device,
            d_model=self.hidden_dim+self.input_len,
            head=1,
            num_nodes=self.num_nodes,
            seq_length=1,
            dropout=0.1,
        )
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim+2*self.input_len, self.hidden_dim+2*self.input_len) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim+2*self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        self.LD = LD(kernel_size=25)

        self.temporal_conv = TemporalConvNet(self.hidden_dim+self.input_len)



    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        raw_data = history_data[..., 0]

        trend = self.LD(raw_data)
        seasonal = raw_data - trend
        trend = trend.unsqueeze(-1)
        seasonal = seasonal.unsqueeze(-1)

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))


        hidden1 = torch.cat([time_series_emb]+tem_emb+node_emb+[seasonal], dim=1)

        tc = self.temporal_conv(hidden1)
        hidden1 = hidden1 + tc
        dat = self.SpatialBlock(hidden1)

        enc_out = hidden1+dat
        hidden = torch.cat([enc_out] + [trend], dim=1)
        hidden = self.encoder(hidden)


        # regression
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        #prediction(64, 12, 170, 1)
        prediction = self.regression_layer(hidden)

        return prediction