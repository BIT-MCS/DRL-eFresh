import os
import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias,use_att=False):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.use_att=use_att

        self.hidden_layer = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=self.bias)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim, bias=self.bias)

        self.num_heads = 8
        self.key_size = 64
        self.value_size = 64
        self.mem_size = hidden_dim
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads

        if self.use_att:
            self.qkv_projector = nn.Linear(self.mem_size, self.total_qkv_size)
            self.qkv_layernorm = nn.LayerNorm([2, self.total_qkv_size])
        else:
            self.input_layer = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim, bias=self.bias)

    def multihead_attention(self, input, rnn_hidden_state):

        memory = torch.stack((input, rnn_hidden_state), dim=1)
        """
        Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """

        # First, a simple linear projection is used to construct queries
        qkv = self.qkv_projector(memory)
        # apply layernorm for every dim except the batch dim
        qkv = self.qkv_layernorm(qkv)

        # mem_slots needs to be dynamically computed since mem_slots got concatenated with inputs
        # example: self.mem_slots=10 and seq_length is 3, and then mem_slots is 10 + 1 = 11 for each 3 step forward pass
        # this is the same as self.mem_slots_plus_input, but defined to keep the sonnet implementation code style
        mem_slots = memory.shape[1]  # denoted as N

        # split the qkv to multiple heads H
        # [B, N, F] => [B, N, H, F/H]
        qkv_reshape = qkv.view(qkv.shape[0], mem_slots, self.num_heads, self.qkv_size)

        # [B, N, H, F/H] => [B, H, N, F/H]
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)

        # [B, H, N, key_size], [B, H, N, key_size], [B, H, N, value_size]
        q, k, v = torch.split(qkv_transpose, [self.key_size, self.key_size, self.value_size], -1)

        # scale q with d_k, the dimensionality of the key vectors
        q *= (self.key_size ** -0.5)

        # make it [B, H, N, N]
        dot_product = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights = torch.softmax(dot_product, dim=-1)

        # output is [B, H, N, V]
        output = torch.matmul(weights, v)

        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        # Residual
        new_memory = new_memory + memory

        return new_memory.view(new_memory.shape[0],-1)

    def forward(self, input_tensor, h_cur):
        if self.use_att:
            combined=self.multihead_attention(input_tensor,h_cur)
        else:
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


class RelationalGRU(nn.Module):
    def __init__(self, input_size, hidden_dim, use_att=False,batch_first=False, bias=True):

        super(RelationalGRU, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.bias = bias
        self.use_att=use_att

        # convert python list to pytorch module
        self.gru_cell = GRUCell(input_dim=self.input_size,
                                hidden_dim=self.hidden_dim,
                                bias=self.bias,
                                use_att=self.use_att)

    def forward(self, input_tensor, hidden_state):
        if self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2)

        seq_len = input_tensor.size(0)
        output_list = []
        for t in range(seq_len):
            output, hidden_state = self.gru_cell(input_tensor=input_tensor[t, :, :],  # (b,t,c)
                                                 h_cur=hidden_state.squeeze(0))

            hidden_state=hidden_state.unsqueeze(0)
            output_list.append(output)

        outputs = torch.stack(output_list, dim=1)

        return outputs, hidden_state

# if __name__ == '__main__':
#     device = torch.device('cuda')
#
#     input_size = 512
#     batch_size = 64
#     time_steps = 10
#     hidden_dim = 512
#     model = RelationalGRU(input_size=input_size, hidden_dim=hidden_dim, use_att=True).to(device)
#
#     input_tensor = torch.rand(time_steps, batch_size, input_size).to(device)  # (t,b,c,h,w)
#
#     hidden_state = torch.zeros(batch_size, hidden_dim).to(device)
#
#     layer_output_list, last_state_list = model(input_tensor, hidden_state)
#     print(layer_output_list.shape, last_state_list.shape)
