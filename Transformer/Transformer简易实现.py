# 在学习Transformer完理论后，对其代码如何实现完全不知。
# 所以利用一个简单的例子来完成Transformer的Pytorch的实现
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch import optim

# 数据部分
# 首先就是数据预处理部分，这里为了训练方便，并没有用什么大型数据集，主要是为了更关注模型本身实现
# 这里的例子为机器翻译例子，手动输入两对德语->英语的句子
sentences = [
        # enc_input                dec_input            dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

src_vocab = {'P':0,'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'S':6,'E':7,'.':8}
tgt_vocab_size = len(tgt_vocab)
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
src_len = 5 #输入句子中最多单词数
tgt_len = 6 #解码器输入（输出）句子中最多单词数

def make_data(sentences):
        enc_inputs,dec_inputs,dec_outputs = [],[],[]
        for i in range(len(sentences)):
                # 将句子中的单词转换成词向量
                enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]# [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
                dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]# [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
                dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]# [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

                enc_inputs.extend(enc_input)
                dec_outputs.extend(dec_output)
                dec_inputs.extend(dec_input)
        return torch.LongTensor(enc_inputs),torch.LongTensor(dec_inputs),torch.LongTensor(dec_outputs)

# 将句子转换成词向量 为接下来的处理做准备
enc_inputs,dec_inputs,dec_outputs=make_data(sentences)

# 创建一个DataSet对象
# 你需要自己定义一个class，里面至少包含3个函数：
# ①__init__：传入数据，或者像下面一样直接在函数里加载数据
# ②__len__：返回这个数据集一共有多少个item
# ③__getitem__:返回一条训练数据，并将其转换成tensor
class MyDataSet(Data.Dataset):
        def __init__(self, enc_inputs, dec_inputs, dec_outputs):
                super(MyDataSet, self).__init__()
                self.enc_inputs = enc_inputs
                self.dec_inputs = dec_inputs
                self.dec_outputs = dec_outputs
        def  __len__(self):
                return self.enc_inputs.shape[0]
        def __getitem__(self, idx):
                return self.enc_inputs[idx],self.dec_inputs[idx],self.dec_outputs[idx]
# 构建DataLoader对象
# 参数：
# dataset：传入的数据
# shuffle = True:是否打乱数据
# collate_fn：使用这个参数可以自己操作每个batch的数据
loader = Data.DataLoader(MyDataSet(enc_inputs,dec_inputs,dec_outputs),2,True)

# 模型参数
d_model = 512 #每个词用多少维表示
d_ff = 2048 # 表示经过两个线性层之前 数据是多少维
d_k= d_v= 64 #表示v，k=q两个矩阵的维度 这里为了方便使v=k
n_layers = 6 # 表示Encoder和Decoder的个数
n_heads = 8 # 表示多头注意力中head的数量

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

# mask码 Pad Mask
def get_attn_pad_mask(seq_q,seq_k):
        """
        :param seq_q:[batch_size,seq_len]
        :param seq_k: [batch_size,seq_len]
        seq_len 无法确定 在Encoder中调用则为src_len
        在Decoder中调用则为src_len也可能为tgt_len
        :return:建议打印出来看看
        """
        batch_size,len_q = seq_q.size()
        batch_size,len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size,len_q,len_k)

# Subsequence Mask
# Subsequence Mask 只有Decoder会用到 主要作用是屏蔽未来时刻单词信息
# 首先np.ones()生成一个全1的方阵，然后通过np.triu()生成一个上三角矩阵
def get_attn_subsequence_mask(seq):
        """
        :param seq: [batch_size,tgt_len]
        :return:
        """
        attn_shape = [seq.size(0),seq.size(1),seq.size(1)]
        subsequence_mask = np.triu(np.ones(attn_shape),k=1)
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask


# 通过Q和K计算出scores 然后将scores和V相乘 得到每个单词的context vector
# 第一步是将Q和K的转置相乘没什么好说的，
# 相乘之后得到的scores还不能立刻进行softmax，
# 需要和attn_mask相加，把一些需要屏蔽的信息屏蔽掉，
# attn_mask是一个仅由True和False组成的tensor，
# 并且一定会保证attn_mask和scores的维度四个值相同
# （不然无法做对应位置相加）
# mask完了之后，就可以对scores进行softmax了。
# 然后再与V相乘，得到context
class ScaledDotProductAttention(nn.Module):
        def __init__(self):
                super(ScaledDotProductAttention, self).__init__()
        def forward(self,Q,K,V,attn_mask):
                '''
                Q: [batch_size, n_heads, len_q, d_k]
                K: [batch_size, n_heads, len_k, d_k]
                V: [batch_size, n_heads, len_v(=len_k), d_v]
                attn_mask: [batch_size, n_heads, seq_len, seq_len]
                '''
                scores = torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
                scores.masked_fill_(attn_mask,-1e9)
                attn = nn.Softmax(dim=-1)(scores)
                context = torch.matmul(attn,V)
                return context,attn


# MultiHeadAttention
class MultiHeadAttention(nn.Module):
        def __init__(self):
                super(MultiHeadAttention, self).__init__()
                self.W_Q = nn.Linear(d_model,d_k*n_heads,bias=False)
                self.W_K = nn.Linear(d_model,d_k*n_heads,bias=False)
                self.W_V = nn.Linear(d_model,d_v*n_heads,bias=False)
                self.fc = nn.Linear(n_heads*d_v,d_model,bias=False)
        def forward(self,input_Q,input_K,input_V,attn_mask):
                residual,batch_size = input_Q,input_Q.size(0)
                Q = self.W_Q(input_Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)
                K = self.W_K(input_K).view(batch_size,-1,n_heads,d_k).transpose(1,2)
                V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
                attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)
                context,attn = ScaledDotProductAttention()(Q,K,V,attn_mask)
                context = context.transpose(1,2).reshape(batch_size,-1,n_heads*d_v)
                output = self.fc(context)
                return nn.LayerNorm(d_model)(output + residual),attn

# FeedForward Layer
class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
                super(PoswiseFeedForwardNet, self).__init__()
                self.fc=nn.Sequential(
                        nn.Linear(d_model,d_ff,bias=False),
                        nn.ReLU(),
                        nn.Linear(d_ff,d_model,bias=False)
                )
        def forward(self,inputs):
                residual = inputs
                output = self.fc(inputs)
                return nn.LayerNorm(d_model)(output + residual)

# Encoder Layer
class EncoderLayer(nn.Module):
        def __init__(self):
                super(EncoderLayer, self).__init__()
                self.enc_self_attn = MultiHeadAttention()
                self.pos_ffn = PoswiseFeedForwardNet()
        def forward(self,enc_inputs,enc_self_attn_mask):
                enc_outputs,attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
                enc_outputs = self.pos_ffn(enc_outputs)
                return enc_outputs,attn

# Encoder
class Encoder(nn.Module):
        def __init__(self):
                super(Encoder, self).__init__()
                self.src_emb = nn.Embedding(src_vocab_size,d_model)
                self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encodingg_table(src_vocab_size,d_model),freeze=True)
                self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        def forward(self,enc_inputs):
                word_emb = self.src_emb(enc_inputs)
                pos_emb = self.pos_emb(enc_inputs)
                enc_outputs = word_emb + pos_emb
                enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)
                enc_self_attns = []
                for layer in self.layers:
                        enc_outputs,enc_self_attn = layer(enc_outputs,enc_self_attn_mask)
                        enc_self_attns.append(enc_self_attn)
                return enc_outputs,enc_self_attns

# Decoder Layer
class DecoderLayer(nn.Module):
        def __init__(self):
                super(DecoderLayer, self).__init__()
                self.dec_self_attn = MultiHeadAttention()
                self.dec_enc_attn = MultiHeadAttention()
                self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
                dec_outputs,dec_self_attn = self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
                dec_outputs,dec_enc_attn = self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
                dec_outputs = self.pos_ffn(dec_outputs)
                return dec_outputs,dec_self_attn,dec_enc_attn

#Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encodingg_table(tgt_vocab_size, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        word_emb = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        pos_emb = self.pos_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = word_emb + pos_emb
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

# Transformer
class Transformer(nn.Module):
        def __init__(self):
                super(Transformer, self).__init__()
                self.encoder = Encoder()
                self.decoder = Decoder()
                self.projection = nn.Linear(d_model,tgt_vocab_size,bias=False)
        def forward(self,enc_inputs,dec_inputs):
                enc_outputs,enc_self_attns = self.encoder(enc_inputs)
                dec_outputs,dec_self_attns,dec_enc_attns = self.decoder(dec_inputs,enc_inputs,enc_outputs)
                dec_logits = self.projection(dec_outputs)
                return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_enc_attns
        

model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.99)


# 训练
for epoch in range(30):
        for enc_inputs,dec_inputs,dec_outputs in loader:
                outputs,enc_self_attns,dec_self_attns = model(enc_inputs,dec_inputs)
                loss = criterion(outputs,dec_outputs.view(-1))
                print("Epoch:","%04d"%(epoch + 1),'loss=','{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
# 测试
enc_inputs,dec_inputs,_ = next(iter(loader))
predict,_,_ = model(enc_inputs[0].view(1,-1),dec_inputs[0].view(1,-1))
predict = predict.data.max(1,keepdim=True)[1]
print(enc_inputs[0],'->',[idx2word[n.item()] for n in predict.squeeze()])











