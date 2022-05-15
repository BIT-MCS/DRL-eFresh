import torch
from torch import nn
from main_setting import Params
import matplotlib.pyplot as plt
import numpy as np
import time
import os

params = Params()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    else:
        print('None bias')
    return module


init_ = lambda m: init(m,
                       nn.init.orthogonal_,
                       lambda x: nn.init.constant_(x, 0),
                       nn.init.calculate_gain('relu'))


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.input_layer = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim, bias=self.bias)
        self.hidden_layer = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=self.bias)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim, bias=self.bias)

        self.num_heads = 8
        self.key_size = 64
        self.value_size = 64
        self.mem_size = hidden_dim
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.input_layer(combined)

        gamma, beta = torch.split(combined, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.hidden_layer(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm

        output = torch.sigmoid(self.output_layer(h_next))
        return output, h_next


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.input_conv_layer = init_(nn.Conv2d(in_channels=input_dim + hidden_dim,
                                                out_channels=2 * self.hidden_dim,
                                                # for update_gate,reset_gate respectively
                                                kernel_size=kernel_size,
                                                padding=self.padding,
                                                bias=self.bias))

        self.hidden_conv_layer = init_(nn.Conv2d(in_channels=input_dim + hidden_dim,
                                                 out_channels=self.hidden_dim,  # for candidate neural memory
                                                 kernel_size=kernel_size,
                                                 padding=self.padding,
                                                 bias=self.bias))

        self.output_conv_layer = init_(nn.Conv2d(in_channels=hidden_dim,
                                                 out_channels=hidden_dim,  # for update_gate,reset_gate respectively
                                                 kernel_size=kernel_size,
                                                 padding=self.padding,
                                                 bias=self.bias))

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.input_conv_layer(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.hidden_conv_layer(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        output = torch.sigmoid(self.output_conv_layer(h_next))
        return output, h_next


class QueryNetwork(nn.Module):
    def __init__(self):
        super(QueryNetwork, self).__init__()
        # TODO: Add proper non-linearity.
        self.model = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 288), nn.ReLU(), nn.Linear(288, 288)
        )

    def forward(self, query):
        out = self.model(query)
        return out.reshape(-1, 4, 72)


class SpatialBasis:
    # TODO: Implement Spatial.
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    def __init__(self, height=8, width=8, channels=64):
        h, w, d = height, width, channels

        p_h = torch.mul(torch.arange(1, h + 1).unsqueeze(1).float(), torch.ones(1, w).float()) * (np.pi / h)
        p_w = torch.mul(torch.ones(h, 1).float(), torch.arange(1, w + 1).unsqueeze(0).float()) * (np.pi / w)

        # NOTE: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values. Still, I think what I have is aligned with what
        # they did, but I am less confident in this step.
        U = V = 8  # size of U, V.
        u_basis = v_basis = torch.arange(1, U + 1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum('hwu,hwv->hwuv', torch.cos(a), torch.cos(b)).reshape(h, w, d)
        self.S = out

    def __call__(self, X):
        # Stack the spatial bias (for each batch) and concat to the input.
        batch_size = X.size()[0]
        S = torch.stack([self.S] * batch_size).to(X.device)
        return torch.cat([X, S], dim=3)


def spatial_softmax(A):
    # A: batch_size x h x w x d
    b, h, w, d = A.size()
    # Flatten A s.t. softmax is applied to each grid (not over queries)
    A = A.reshape(b, h * w, d)
    A = torch.softmax(A, dim=1)
    # Reshape A to original shape.
    A = A.reshape(b, h, w, d)
    return A


def apply_alpha(A, V):
    # TODO: Check this function again.
    b, h, w, c = A.size()
    A = A.reshape(b, h * w, c).transpose(1, 2)

    _, _, _, d = V.size()
    V = V.reshape(b, h * w, d)

    return torch.matmul(A, V)


class SpatialAttGRU(nn.Module):
    def __init__(self, input_dim, batch_first=False):
        super(SpatialAttGRU, self).__init__()
        self.batch_first = batch_first
        self.input_conv_dim = input_dim

        self.conv = nn.Sequential(
            # input: 3*80*80
            init_(nn.Conv2d(self.input_conv_dim, 32, 8, stride=4)),
            nn.LayerNorm([32, 19, 19]),
            nn.ReLU(inplace=True),
            # input: 32*19*19
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(inplace=True),
            # input: 64*8*8
        )

        # convert python list to pytorch module
        self.vision_core = ConvGRUCell(input_size=(8, 8),
                                       input_dim=64,
                                       hidden_dim=128,
                                       kernel_size=3)

        # Attention
        self.k_size = 8
        self.v_size = 120
        self.s_size = 64
        self.num_queries = 4

        self.query = QueryNetwork()
        self.spatial = SpatialBasis()

        self.answer_processor = nn.Sequential(
            # 1026 x 512
            # TODO:prev reward action
            nn.Linear(
                (self.v_size + self.s_size) * self.num_queries + (self.k_size + self.s_size) * self.num_queries,
                512
            ),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        self.policy_core = GRUCell(256, 256)
        self.plot_id = 0

    def forward(self, input_tensor, temporal_hidden_state, spatial_hidden_state):
        if self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        seq_len = input_tensor.size(0)
        output_list = []
        for t in range(seq_len):
            # 1 (a). Vision.
            # --------------
            conv_input = self.conv(input_tensor[t, :, :, :, :])
            conv_output, spatial_hidden_state = self.vision_core(input_tensor=conv_input,  # (b,t,c,h,w)
                                                                 h_cur=spatial_hidden_state.squeeze(0))
            spatial_hidden_state = spatial_hidden_state.unsqueeze(0)
            conv_output = conv_output.permute(0, 2, 3, 1)
            K, V = conv_output.split([self.k_size, self.v_size], dim=3)
            K, V = self.spatial(K), self.spatial(V)

            # 1 (b). Queries.
            # --------------
            Q = self.query(temporal_hidden_state)

            # 2. Answer.
            # ----------
            # (n, h, w, num_queries)
            A = torch.matmul(K, Q.transpose(2, 1).unsqueeze(1))
            # (n, h, w, num_queries)
            A = spatial_softmax(A)

            if params.debug_mode:
                att_plot_path = os.path.join(params.root_path, "att_plot")
                os.makedirs(att_plot_path,exist_ok=True)
                A_sum=torch.sum(A,dim=(0,3))
                plt.matshow(A_sum.numpy())
                plt.savefig(att_plot_path + "/%d step_att.png" %self.plot_id)
                plt.close()
                self.plot_id += 1

            # (n, 1, 1, num_queries)
            a = apply_alpha(A, V)

            # (n, (c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + 1)
            # TODO:prev reward,action
            answer = torch.cat(torch.chunk(a, 4, dim=1) + torch.chunk(Q, 4, dim=1), dim=2, ).squeeze(1)
            # (n, hidden_size)
            answer = self.answer_processor(answer)
            temporal_output, temporal_hidden_state = self.policy_core(input_tensor=answer,
                                                                      h_cur=temporal_hidden_state.squeeze(0))

            temporal_hidden_state = temporal_hidden_state.unsqueeze(0)
            output_list.append(temporal_output)

        outputs = torch.stack(output_list, dim=1)

        return outputs, temporal_hidden_state, spatial_hidden_state

# if __name__ == '__main__':
#     device = torch.device('cuda')
#
#     height = width = 80
#     hidden_height = hidden_width = 8
#     channels = 2
#     batch_size = 128
#     time_steps = 10
#     spatial_hidden_dim = 128
#     temporal_hidden_dim=256
#
#     model = SpatialAttGRU(input_dim=channels).to(device)
#
#     input_tensor = torch.rand(time_steps, batch_size, channels, height, width).to(device)  # (b,t,c,h,w)
#
#     spatial_hidden_state = torch.zeros(batch_size, spatial_hidden_dim, hidden_height, hidden_width).to(device)
#
#     temporal_hidden_state = torch.zeros(batch_size,temporal_hidden_dim).to(device)
#
#     layer_output_list, last_temporal_state, last_spatial_state = model(input_tensor, temporal_hidden_state,spatial_hidden_state)
#
#     print(layer_output_list.shape,last_temporal_state.shape,last_spatial_state.shape)
