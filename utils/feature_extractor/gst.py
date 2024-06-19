import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# 多头注意力机制-在查询序列与键序列之间计算注意力分数，将这些分数应用于值序列以生成加权和。
# 用于建立音频的时间序列之间的关联性
class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        # num_heads 的输出维度
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        # 注意力分数
        scores = torch.matmul(querys, keys.transpose(2, 3)).contiguous()  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        # 获取注意力权重
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values).contiguous()  # [h, N, T_q, num_units/h]
        # 拼接多头注意力维度
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        return out

# 从输入的梅尔频谱图中提取特征
# 包括一系列的卷积层（通过不同的卷积核尺寸和步幅来逐渐减小时间分辨率），之后经过一个GRU层，最终输出一个固定大小的特征向量
class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self):
        super().__init__()
        ref_enc_filters = [32, 32, 64, 64, 128, 128]    # 卷积层输出通道
        n_mel_channels = 80     # mel频谱的通道数
        ref_enc_gru_size = 128      # GRU（门控循环单元）的隐藏状态维度_128维
        K = len(ref_enc_filters)        # 卷积层的数量-与 ref_enc_filters 卷积层输出通道的长度相同。
        filters = [1] + ref_enc_filters     # 卷积层的输入通道数

        # 卷积层
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        # 批归一化层
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i])
             for i in range(K)])
        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        # GRU门控循环单元层
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = n_mel_channels
        self.ref_enc_gru_size = ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        # 确保输入的特征表示的通道数等于预设的 n_mel_channels，即mel帧数正确。
        assert inputs.size(-1) == self.n_mel_channels      # 80！
        #
        out = inputs.unsqueeze(0)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)   # 卷积层
            out = bn(out)       # 批归一化层
            out = F.relu(out)       # relu激活函数
        # 将处理后的结果在第一个和第二个维度进行转置
        # contiguous() 函数使得结果张量在内存中是连续存储的。
        out = out.transpose(1, 2).contiguous()  # [N, Ty//2^K, 128, n_mels//2^K]
        # 获取维度
        N, T = out.size(0), out.size(1)
        # 将转置后的结果进行扁平化，使得结果具有形状 [N, T, 128 * n_mels // 2^K]
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            # print(input_lengths.cpu().numpy(), 2, len(self.convs))
            # 将输入序列的长度从 PyTorch 张量转换为 NumPy 数组，并按照卷积层数进行调整。
            input_lengths = (input_lengths.cpu().numpy() / 2 ** len(self.convs))
            # 将长度进行四舍五入并转换为整数，确保长度至少为 1。
            input_lengths = max(input_lengths.round().astype(int), [1])
            # print(input_lengths, 'input lengths')
            # 将处理后的结果按照输入序列的长度进行动态序列打包，以便在 RNN（GRU）中使用。
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths, batch_first=True, enforce_sorted=False)
        # 将 GRU 模型的参数展平，以提高性能
        self.gru.flatten_parameters()
        # 返回 GRU 的输出和隐藏状态。
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, l, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            l = (l - kernel_size + 2 * pad) // stride + 1
        return l

# Style Token Layer
# 计算样式标记（Style Token）。
# 将输入嵌入为一个低维度的样式标记，并应用自注意力机制来捕捉输入序列中的关联信息。这些样式标记用于表示音频中的风格信息
class STL(nn.Module):
    """
    inputs --- [N, token_embedding_size//2]
    """

    def __init__(self, token_embedding_size):
        super().__init__()
        token_num = 10
        num_heads = 8
        self.embed = nn.Parameter(torch.FloatTensor(token_num, token_embedding_size // num_heads))
        d_q = token_embedding_size // 2
        d_k = token_embedding_size // num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=token_embedding_size,
            num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        # 获取输入样式向量的批次大小
        N = inputs.size(0)
        # 查询向量 query，用于注意力计算。
        query = inputs.unsqueeze(1)
        # 通过 tanh 函数将样式令牌的嵌入进行非线性变换，再扩展为与输入批次大小相同的形状，
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size//num_heads]
        # print(query.shape, keys.shape)
        # 使用多头注意力机制计算样式嵌入
        style_embed = self.attention(query, keys)

        return style_embed

# 核心（将上述三个模型组合到一起MultiHeadAttention、ReferenceEncoder、STL）
# 首先，通过 ReferenceEncoder 提取输入音频的特征。然后，通过 STL 计算全局风格特征。最后，通过 categorical_layer 对全局风格特征进行分类，输出不同风格类别的概率分布。
class GST(nn.Module):
    def __init__(self, token_embedding_size, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = STL(token_embedding_size)

        self.categorical_layer = nn.Linear(token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        # 将输入信息通过ReferenceEncode编码得到特征表示
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        # print(enc_out.shape)
        # 特征信息通过STL模型获取风格表示
        style_embed = self.stl(enc_out)
        # 压缩为第一个维度，并通过线性层映射为不同类别的概率分布，然后使用 softmax 函数将概率进行归一化
        cat_prob = F.softmax(self.categorical_layer(style_embed.squeeze(0)), dim=-1)
        # print(style_embed.shape, cat_prob.shape)
        # 返回风格特征和分类概率
        # style_embed.squeeze(0)为适应输出的形状要求
        return (style_embed.squeeze(0), cat_prob)


# load_npy = np.load("E:\\dialog_TTS\\DailyTalk_interspeech23\\preprocessed_data\\DailyTalk\\mel_frame\\1-mel-5_1_d2136.npy")
# mel = torch.tensor(load_npy).unsqueeze(0)
# # print(load_npy)

# gst = GST(256,7)
# out = gst(mel)
# print(out[0].shape)
# print(out[1])