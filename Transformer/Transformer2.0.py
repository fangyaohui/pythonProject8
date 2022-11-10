# 在学习Transformer完理论后，对其代码如何实现完全不知。
# 所以利用一个简单的例子来完成Transformer的Pytorch的实现
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# S: 表示decoding输入的开始符号
# E: 表示decoding输出结束符号
# P: 如果当前批次数据大小小于时间步长，将填充空白序列的符号
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
# 手动构建词向量 注意Decoder和Encoder的单词不能放在一起构建词向量
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab) # 表示Encoder词向量的个数

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}   # Decoder的词向量
idx2word = {i: w for i, w in enumerate(tgt_vocab)}  # 构建字典 根据词向量映射单词
tgt_vocab_size = len(tgt_vocab) # Decoder词向量的个数

src_len = 5  # 表示Encoder输入的一个句子中最多的单词个数
tgt_len = 6  # 表示Decoder输入（输出）的一个句子中最多的单词个数

# Transformer Parameters Transformer参数 重要参数
d_model = 512  # 字嵌入&位置嵌入的维度，这俩值是相同的，因此用一个变量就行了 表示词向量的维度以及位置向量的维度
# FeedForward dimension 表示Feed Forward隐藏层神经元的个数
d_ff = 2048
# Q、K、V向量的维度，其中Q与K的维度必须相等，
# V的维度没有限制，不过为了方便起见，都设为64
d_k = d_v = 64  # dimension of K(=Q), V
# Encoder和Decoder的个数
n_layers = 6
# 多头注意力中head的数量
n_heads = 8

"""
make_data(sentences)函数
参数：sentences为Encoder要输入的数据（包括Encoder输入 Decoder输入 Decoder输出）
函数作用：将每个部分分词 根据构建的单词对应字典 将单词转换成对应的词向量。
注意将Encoder Deconder 不同的部分拆开来赋值
return：将得到的矩阵转化成Tensor 使用torch.LongTensor(xxx)进行转化
"""
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

# 得到各个部分对应的词向量
"""
enc_inputs:Encoder输入部分
dec_inputs:Decoder输入部分
dec_outputs:Decoder输出部分
"""
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

"""
创建dataset对象 继承Data.Dataset 
必须包含__init__,__len__和__getitem__函数

"""
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# 创建dataloader对象
# 这里batch_size=2
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

# Positional Encoding
# 参数：n_position字库的大小 d_model位置编码的维度
def get_sinusoid_encodingg_table(n_position,d_model):
        def cal_angle(position,hid_idx):
                return position/np.power(10000,2*(hid_idx//2)/d_model)
        def get_posi_angle_vec(position):
                return [cal_angle(position,hid_j) for hid_j in range(d_model)]
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2]) # dim 2i
        sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2]) # dim 2i+1
        return torch.FloatTensor(sinusoid_table)

# 实现padding mask
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    由于在Encoder和Decoder中都需要进行mask操作，
    因此就无法确定这个函数的参数中seq_len的值，
    如果是在Encoder中调用的，seq_len就等于src_len；
    如果是在Decoder中调用的，seq_len就有可能等于src_len，
    也有可能等于tgt_len（因为Decoder有两次mask）
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
# 实现sequence_mask

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 实现上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]
