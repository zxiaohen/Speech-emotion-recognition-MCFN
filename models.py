
import torch.nn.functional as F
from Dual_attention import *
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class AdditiveAttention(nn.Module):

   def __init__(self, key_size, query_size, num_hiddens, **kwargs):
       super(AdditiveAttention, self).__init__(**kwargs)
       self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
       self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
       self.w_v = nn.Linear(num_hiddens, 1, bias=False)

   def forward(self, queries, keys, values):

       queries, keys = self.W_q(queries), self.W_k(keys)   #(queries~[2, 1, 8],keys~[2, 10, 8])
       features = queries.unsqueeze(2) + keys.unsqueeze(1)   # features~[2, 1, 10, 8]
       features = torch.tanh(features)  #[2, 1, 10, 8]
       scores = self.w_v(features).squeeze(-1)
       self.attention_weights =nn.functional.softmax(scores,dim=-1)  # (attention_weights~[2,1,10])
       return torch.bmm(self.attention_weights, values.transpose(1,2))

#bimodal model for audio-text feature extraction
class model_bimodal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.semantic_embed = torch.load('D:/appstorage/PyCharm Community Edition 2022.2.1/fine_grained/checkpoint/bert_best2.pt')
        self.semantic_embed.config.output_hidden_states = True
        for param in self.semantic_embed.parameters():
            param.requires_grad = False
       # print(self.semantic_embed)
        self.semantic_embed.eval()
        self.semantic_lstm = nn.LSTM(768, 128, 1, bidirectional=True, batch_first=True, dropout=0.5)
        self.semantic_linear1 = nn.Linear(768,512 )
        self.semantic_linear2 = nn.Linear(512, 128)# This the embedding for the audio features

        self.pre_cnn = nn.Conv1d(config['acoustic']['embedding_dim'], config['acoustic']['embedding_dim'], 3, 1,
                                 padding=1)
        self.acoustic_cnn = nn.Conv1d(config['acoustic']['embedding_dim'], 256, 5, 1, dilation=2)
        self.bn = nn.BatchNorm1d(256, affine=False)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'], 64, 3, 1, dilation=2)
        self.bn1 = nn.BatchNorm1d(64, affine=False)
        self.m1 = nn.LeakyReLU(0.01)
        self.acoustic_cnn2 = nn.Conv1d(64, 128, 2, 1, dilation=2)
        self.bn2 = nn.BatchNorm1d(128, affine=False)
        self.acoustic_cnn3 = nn.Conv1d(128, 256, 2, 1, dilation=2)
        self.bn3 = nn.BatchNorm1d(256, affine=False)
        # self.acoustic_cnn3 = nn.Conv1d(128, 256, 2, 1)
        self.acoustic_mean1 = nn.AvgPool2d((1, 5), (1, 1))
        self.acoustic_mean2 = nn.AvgPool2d((1, 3), (1, 1))
        self.acoustic_mean3 = nn.AvgPool2d((1, 3), (1, 1))
        self.fuse_lstm = nn.LSTM(512, 256, 1, bidirectional=True,
                                 batch_first=True, dropout=0.5)
        # Add the cross-modal excitement layer
        self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size'] + 1, 256)
        self.semantic_excit = nn.Linear(256, 256)
        self.classifier = nn.Linear(2 * 256, 4)
        self.attention = AdditiveAttention(256, 256, 128)
        self.mha = MultiHeadAttention(key_size=256, query_size=256, value_size=256, num_hiddens=256, num_heads=4)
        self.pa = TAM_Module(256)
        self.ca = CAM_Module(256)
        self.conv_postatt = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1, bias=False), nn.BatchNorm1d(256, eps=1e-6),
                                          nn.ReLU())
        self.conv_final = nn.Sequential(nn.Dropout1d(0.1, False), nn.Conv1d(256, 256, 1))

        self.pa_a = TAM_Module(256)
        self.ca_a = CAM_Module(256)
        self.conv_postatt_a = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1, bias=False),
                                            nn.BatchNorm1d(256, eps=1e-6),
                                            nn.ReLU())
        self.conv_final_a = nn.Sequential(nn.Dropout1d(0.1, False), nn.Conv1d(128, 128, 1))


    def forward(
            self,
            acoustic_input,
            acoustic_length,
            asr_text,
            semantic_input,
            semantic_input_model,
            semantic_length,):

        hidden_states=self.semantic_embed(asr_text["input_ids"]).hidden_states
        semantic_embed=hidden_states[-1]
        semantic_embed,_ = self.semantic_lstm(semantic_embed)

        semantic_embed_new = semantic_embed.transpose(1, 2)
        pa_out = self.pa(semantic_embed_new)
        pa_out = self.conv_postatt(pa_out)
        ca_out = self.ca(semantic_embed_new)
        ca_out = self.conv_postatt(ca_out)
        feat_sum = pa_out + ca_out
        semantic_embed = feat_sum.transpose(1, 2)

        acoustic_embed = self.pre_cnn(acoustic_input.permute(0, 2, 1))

        acoustic_embed = self.m1(acoustic_embed)
        acoustic_para = self.acoustic_cnn(acoustic_embed)

        acoustic_embed = self.acoustic_cnn1(acoustic_embed)
        acoustic_para = self.bn(acoustic_para)
        acoustic_embed = self.bn1(acoustic_embed)
        acoustic_embed = self.m1(acoustic_embed)

        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_embed = self.bn2(acoustic_embed)
        acoustic_embed = self.m1(acoustic_embed)
        acoustic_embed = self.acoustic_cnn3(acoustic_embed)  # [B,C,A]
        # print(acoustic_embed.shape)
        acoustic_embed = self.bn3(acoustic_embed)
        acoustic_embed = self.m1(acoustic_embed)
        acoustic_embed = acoustic_embed + acoustic_para
        acoustic_embed = self.m1(acoustic_embed)
        #
        acoustic_embed_new = acoustic_embed
        # print(acoustic_embed.shape)
        pa_out_a = self.pa_a(acoustic_embed_new)
        pa_out_a = self.conv_postatt_a(pa_out_a)
        ca_out_a = self.ca_a(acoustic_embed_new)
        ca_out_a = self.conv_postatt_a(ca_out_a)
        feat_sum_a = pa_out_a + ca_out_a
        # feat_sum=self.conv_final(feat_sum)
        acoustic_embed = feat_sum_a
        mha_out = self.mha(semantic_embed, acoustic_embed.transpose(1, 2), acoustic_embed.transpose(1, 2))
        acoustic_embed = mha_out


        semantic_excit = F.sigmoid(self.semantic_excit(acoustic_embed))
        semantic_embed = semantic_embed * semantic_excit + semantic_embed
        # acoustic_excit = F.sigmoid(self.acoustic_excit(semantic_input))
        # acoustic_embed = acoustic_embed * acoustic_excit + acoustic_embed  # These two lines are different, we add the residual connection
        #print(semantic_embed.shape,acoustic_embed.shape)
        fuse_embed = torch.cat([semantic_embed, acoustic_embed], dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed,  semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)

        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        fuse_embed = torch.max(fuse_embed, dim=1)[0]
        logits = self.classifier(fuse_embed)
        return logits,fuse_embed

#unimodal model for audio feature extraction
class model_acoustic(nn.Module):
    def __init__(self, config, feature_type='acoustic'):
        super().__init__()
        self.feature_type = feature_type
        self.acoustic_lstm = nn.LSTM(
                config['acoustic']['embedding_dim'],
                config['acoustic']['hidden_dim'], 1, bidirectional=True,
                batch_first=True, dropout=0.5 )
        self.classifier = nn.Linear(
                2 * config['acoustic']['hidden_dim'],
                config['classifier']['class_num'] )
        self.attention = nn.Parameter(
                torch.randn(2 * config['acoustic']['hidden_dim']) )
        self.multihead_attn = nn.MultiheadAttention(2 * config['acoustic']['hidden_dim'], 8,batch_first=True)
        self.loss_name = config['loss']['name']

    def forward(
            self,
            acoustic_input,
            acoustic_length,
            semantic_input,
            semantic_length, ):

        acoustic_pack = nn.utils.rnn.pack_padded_sequence(
               acoustic_input, acoustic_length.cpu(), batch_first=True, enforce_sorted=False)
        acoustic_embed, (h_n, c_n) = self.acoustic_lstm(acoustic_pack)
        acoustic_embed, _ = nn.utils.rnn.pad_packed_sequence(acoustic_embed, batch_first=True)  # [B,A,D]

        attention_mask1 = ~torch.tril(torch.ones([acoustic_input.size(1),acoustic_input.size(1)])).bool().to(acoustic_input.device)
        attn_output, attn_output_weights = self.multihead_attn(acoustic_embed, acoustic_embed, acoustic_embed,
                                                               attn_mask=attention_mask1)
        lo=torch.mean(self.classifier(attn_output),dim=1)
        attention_mask = torch.arange(
                acoustic_input.size(1))[None, :].repeat(acoustic_input.size(0), 1
                                                        ).to(acoustic_input.device)
        attention_mask = (attention_mask < acoustic_length[:, None].repeat(1, acoustic_input.size(1))).float()[:, :,None]  # [B,A,1]

        attention_score = torch.matmul(acoustic_embed,self.attention[None, :, None].repeat(acoustic_input.size(0), 1, 1))  # [B,A,1]

        attention_score = attention_score / np.sqrt(self.attention.size(0))

        attention_score = attention_score * attention_mask - 1e6 * (1 - attention_mask)
        attention_score = F.softmax(attention_score, dim=1)
        acoustic_embed = torch.matmul(attention_score.permute(0, 2, 1), acoustic_embed).squeeze(1)  # [B,D]

        logits = self.classifier(acoustic_embed)

        return logits,acoustic_embed


# unimodal model for text feature extraction
class model_semantic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.semantic_embed = torch.load('D:/appstorage/PyCharm Community Edition 2022.2.1/fine_grained/checkpoint/bert_best2.pt')
        self.semantic_embed.config.output_hidden_states = True
        for param in self.semantic_embed.parameters():
            param.requires_grad =False
        self.semantic_embed.eval()
        self.semantic_lstm = nn.LSTM(768,128, 2, bidirectional=True,batch_first=True, dropout=0.5)
        self.attention = nn.Parameter(torch.randn( config['semantic']['hidden_dim']))
        self.classifier = nn.Linear(config['semantic']['hidden_dim'], config['classifier']['class_num'])


    def forward(
            self,
            acoustic_input,
            acoustic_length,
            asr_text,
            semantic_input,
            semantic_input_model,
            semantic_length,):

        hidden_states = self.semantic_embed(asr_text["input_ids"]).hidden_states

        semantic_embed = hidden_states[-1]
        semantic_embed,_ = self.semantic_lstm(semantic_embed)

        attention_mask = torch.arange(semantic_input.size(1))[None, :].repeat(semantic_input.size(0), 1).to(semantic_input.device)
        attention_mask = (attention_mask < semantic_length[:, None].repeat(1, semantic_input.size(1))).float()[:, :,None]
        attention_score = torch.matmul(semantic_embed, self.attention[None, :, None].repeat(semantic_input.size(0), 1,1))  # [B,A,1]
        attention_score = attention_score / np.sqrt(self.attention.size(0))

        attention_score = attention_score * attention_mask - 1e6 * (1 - attention_mask)
        attention_score = F.softmax(attention_score, dim=1)
        semantic_embed = torch.matmul(attention_score.permute(0, 2, 1), semantic_embed).squeeze(1)  # [B,D]

        logits = self.classifier(semantic_embed)
        return logits,semantic_embed
