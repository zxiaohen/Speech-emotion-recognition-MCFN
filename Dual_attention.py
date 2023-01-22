import torch
import torch.nn as nn

class AdditiveAttention2(nn.Module):

   def __init__(self, key_size, query_size, num_hiddens, **kwargs):
       super(AdditiveAttention2, self).__init__(**kwargs)
       self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
       self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
       self.w_v = nn.Linear(num_hiddens, 1, bias=False)

   def forward(self, queries, keys, values):

       queries, keys = self.W_q(queries), self.W_k(keys)   #(queries~[2, 1, 8],keys~[2, 10, 8])
       features = queries.unsqueeze(2) + keys.unsqueeze(1)   # features~[2, 1, 10, 8]
       features = torch.tanh(features)  #[2, 1, 10, 8]
       scores = self.w_v(features).squeeze(-1)
       self.attention_weights =nn.functional.softmax(scores,dim=-1)  # (attention_weights~[2,1,10])

       return torch.bmm(self.attention_weights, values)


class MultiHeadAttention(nn.Module):
   def __init__(self, query_size,key_size,  value_size, num_hiddens,num_heads, bias=False, **kwargs):
       super(MultiHeadAttention, self).__init__(**kwargs)
       self.num_heads = num_heads
       self.attention = AdditiveAttention2(int(num_hiddens/num_heads),int(num_hiddens/num_heads), 40)
       self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
       self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
       self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
       self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
   def forward(self, queries, keys, values):
       queries = transpose_qkv(self.W_q(queries), self.num_heads)
       keys = transpose_qkv(self.W_k(keys), self.num_heads)
       values = transpose_qkv(self.W_v(values), self.num_heads)
       output = self.attention(queries, keys, values)
       output_concat = transpose_output(output, self.num_heads)
       return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
   X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
   X = X.permute(0, 2, 1, 3)
   return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
   X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
   X = X.permute(0, 2, 1, 3)
   return X.reshape(X.shape[0], X.shape[1], -1)


class TAM_Module(nn.Module):
    """ Temporal attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(TAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv =nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height= x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1,height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height)
        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height= x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height)
        out = self.gamma*out + x
        return out