from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import jieba

USE_CUDA = torch.cuda.is_available() # pytorch能否使用gpu，gpu训练速度更快，但是对于gpu的内存相比cpu要小，个人电脑可能训练不了大的神经网络
# USE_CUDA = False

# 设置随机种子
random.seed(911001)
np.random.seed(911001)
torch.manual_seed(911001)

if USE_CUDA:
    torch.cuda.manual_seed(911001)

# 超参设置
BATCH_SIZE = 32                # critical：一个batch大小，代表训练的每个step输入BATCH_SIZE个句子到模型中。一般来说，提高BATCH_SIZE有助于提高模型性能
EMBEDDING_SIZE = 128        # critical：词嵌入向量大小
MAX_VOCAB_SIZE = 50000        # critical：字典大小，代表有多少个词/字

GRAD_CLIP = 1               # 梯度裁剪，防止模型的backward的梯度过大，导致模型训偏
NUM_EPOCHS = 100             # critical：训练的轮次

words = set()
word2id = {"<pad>":0}
id2word = {0: "<pad>"}
lines = None

# critical：读数据集并预处理
with open("train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [line.strip('\n') for line in lines if len(line.strip('\n'))]
    lines = [list(jieba.cut(line, cut_all=False)) for line in lines]
    for line in lines:
        for word in line:
            words.add(word)

for i, word in enumerate(list(words)):
    word2id[word] = i+1
    id2word[i+1] = word

# critical：定义词表大小
VOCAB_SIZE = len(word2id)

# critical：定义数据集
class News_Dataset(Dataset):
    def __init__(self, word2id:dict, id2word:dict, texts:List[List[str]]) -> None:
        super().__init__()
        self.word2id = word2id
        self.id2word = id2word
        
        process_text = []
        for text in texts:
            process_text.append(self.convert_words_to_ids(text))
        self.texts = process_text

    def convert_ids_to_words(self, input_ids:List[int]) -> str:
        text = [self.id2word[id] for id in input_ids]
        return ''.join(text)

    def convert_words_to_ids(self, words:List[str]) -> List[int]:
        text = [self.word2id[word] for word in words]
        return text

    def __len__(self):
        return len(self.texts)

    # critical：提供输入和标签
    # 思考1：target为什么取输入sent的[1:]，0又代表了什么？
    # 思考2：当BATCH_SIZE>1时，多个输入的句子可能不等长，
    # 不等长的句子forward会报错，应该怎么处理？
    def __getitem__(self, idx):
        sent = self.texts[idx]
        target = sent[1:] + [0]
        sent = torch.tensor(sent)
        target = torch.tensor(target)
        return sent, target    

# critical：定义训练集，验证集，以及读取训练/验证集的迭代器dataloader
news_ds = News_Dataset(word2id, id2word, lines)
train_dataloader = DataLoader(news_ds)
with open("eval.txt", "r", encoding="utf-8") as f:
    e_lines = f.readlines()
    e_lines = [line.strip('\n') for line in e_lines if len(line.strip('\n'))]
    e_lines = [list(jieba.cut(line, cut_all=False)) for line in e_lines]

eval_ds = News_Dataset(word2id, id2word, e_lines)
eval_dataloader = DataLoader(eval_ds)

# critical：模型的定义
class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, dropout = 0.5):
        super(RNNModel, self).__init__()

        # 定义dropout函数，dropout是一项抑制神经网络过拟合的常见操作
        # 有时候神经网络学得太快也不是好事，因此dropout通过随机丢弃一些神经元的输出，防止网络快速拟合
        # 分布满足伯努利分布，每个神经元有p的概率在一次
        # forward中其参数值计算设置为0
        self.drop = nn.Dropout(dropout)

        # critical：编码器，词从多维映射成低维的
        # embedding为语言模型的第一层，将人能理解的one-hot表示的字词转化为人看不懂，机器看的懂的hidden state\
        self.encoder = nn.Embedding(ntoken, ninp)

        # 设置模型，LSTM和GRU使用nn.LSTM, nn.GRU
        # LSTM和GRU都是改进版的RNN，不用太在意他们的具体实现
        # 若想了解，可以百度一下
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp,
                nhid, dropout = dropout, batch_first=True)

        # 使用普通RNN
        else:
            # 为RNN选择激活函数，tanh, relu对应RNN的激活函数
            # 激活函数本质是一些非线性的运算，他们存在的目的就是为神经网络带来非线性的运算
            # 否则当神经网络只有矩阵乘法时，他们和普通的线性函数没有任何区别，学习和表征能力也大大下降
            try:
                nonlinearity = {"RNN_TANH":'tanh', "RNN_RELU":'relu'}[rnn_type]
            except KeyError:
                raise ValueError("unknown parameter, you can use ['LSTM', \
                    'GRU', 'RNN_TANH', 'RNN_RELU']")

            # critical：模型的核心模块，RNN模型，阅读pytorch文档了解RNN类的定义和输入输出
            self.rnn = nn.RNN(ninp, nhid, nonlinearity = nonlinearity, dropout = dropout, batch_first=True)

        # critical：解码器，目的为将RNN输出的hidden_state（维度为nhid），经过一个线性变换后，
        # 计算出下一个字词的概率（这里并不直接算出概率，而是为词表每个字词算分数），维度为ntoken
        # 线性变化, 定义为y = A * x + b
        # weight的size大小为(nhid, ntoken), 初始化值是u(-1/nhid, 1/nhid)
        # bias的size大小为(ntoken), 初始化值是u(-1/ntoken, 1/ntoken) 
        self.decoder = nn.Linear(nhid, ntoken)

        # 初始化权值,如果不初始化，所有神经元默认都为0
        # 这将大大影响初始的训练速度
        self.init_weights()

        # 记录值，rnn_type，nhid大小，层数, 词表大小
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.vocab_size = ntoken

    def init_weights(self):
        initrange = 0.1
        # 词嵌入模型的权值初始化
        self.encoder.weight.data.uniform_(-initrange, initrange)

        # 线性转化器的权值和偏移值大小
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # critical：定义前向传播
    def forward(self, inputs, hidden):
        
        # 输入编码之后传入dropout计算
        emb = self.drop(self.encoder(inputs))

        # critical：rnn 运行模型的前向传播
        # 以句子[我 爱 中 山 大 学]为例
        # hidden：最后一次的输出，即输入[学]后的输出，会被decoder直接使用
        # output：每一次输入的输出，包含输入[我] [我爱]...[我爱中山大学]后的输出
        output, hidden = self.rnn(emb)

        # 得出值进行dropout
        output = self.drop(output)

        # critical：然后进行线性转换，view函数的用处是改变tensor的形状
        # 因为我们要训练，所以对每个位置的字词都要预测一次，所以使用包含所有字词的output
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    # 初始化隐层值
    def init_hidden(self, bsz, requires_grad = True):
        weight = next(self.parameters())

        # print(list(iter(self.parameters())))

        if self.rnn_type == "LSTM":
            return (weight.new_zeros((1, bsz, self.nhid), requires_grad = requires_grad),
                weight.new_zeros((1, bsz, self.nhid), requires_grad = requires_grad))

        else:
            return weight.new_zeros((1, bsz, self.nhid), requires_grad = requires_grad)

nhid = 128 # critical：模型hidden state的维度，一般而言hidden state越大模型的效果越好，但是训练的速度和需要的数据也越多
model = RNNModel('RNN_RELU', VOCAB_SIZE, EMBEDDING_SIZE,
    nhid, dropout = 0.5)

# 如果USE_CUDA为True，则使用gpu训练模型，将模型从cpu迁移到gpu
if USE_CUDA:
    model = model.cuda()

# model.init_hidden(BATCH_SIZE)
# model

# critical：模型评估，建议先跳过这部分最后再看
def evaluate(model, dataloader):
    # 进入评估状态
    model.eval()
    total_loss = 0
    total_count = 0

    # 不是训练，关闭梯度加快运行速度
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad = False)
        # 将数据按batch输入
        for i, batch in enumerate(dataloader):
            data, target = batch
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()

            hidden = repackage_hidden(hidden)

            with torch.no_grad():
                output, hidden = model(data, hidden)

                ##### model(data,hidden) 相当于调用model.forward

            # 计算损失
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))

            total_count += np.multiply(*data.size())

            total_loss += loss.item() * np.multiply(*data.size())

        loss = total_loss / total_count
        model.train()

        return loss

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# critical：定义损失函数，一般使用交叉熵损失
# 建议阅读pytorch文档了解细节
loss_fn = nn.CrossEntropyLoss()

# critical：学习率，对模型的学习非常重要的超参数（超参数即需要人为设定的参数）
# 过大的学习率会导致训崩，过小的学习率会导致学不动
learning_rate = 0.001

# critical：优化器，更新模型参数的东西
# 优化器需要设置"要被更新的参数"和”更新参数的学习率”
# 默认使用Adam即可，它是一种更先进精良的SGD算法
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# 学习率的优化，进阶的优化形式，此处不用
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

# critical：训练的主要流程
val_losses = []
for epoch in range(NUM_EPOCHS):
    # critical：训练前务必手动设置model.train()
    # model.train()对应另一个函数model.eval(),前者有梯度用于训练，后者无梯度节省内存用于测试
    # 神经网络的后向传播和更新都依赖于梯度，没有梯度跑几个epoch都是无济于事
    model.train()
    hidden = model.init_hidden(BATCH_SIZE)
    
    # critical：将数据集中的数据按batch_size划分好，一一读入模型中
    for i, batch in enumerate(train_dataloader):
        data, target = batch

        # 使用gpu训练需要将数据也迁移到gpu
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        hidden = repackage_hidden(hidden)
        model.zero_grad() # critical：每步运行之前清空前一步backward留下的梯度，否则梯度信息不准确

        # print(data.size(), hidden[0].size())
        # critical：模型的forward，将数据正式传入模型中计算并输出结果
        # 输入：hidden：[BATCH_SIZE, seq_max_len]
        output, hidden = model(data, hidden)

        # critical：计算模型输出与真实标签的差距，也就是损失loss
        # 需要注意，设计模型时没有必要对output进行手动softmax为概率分布
        # nn.CrossEntropyLoss()会自动帮你完成这一步，否则二次softmax将导致模型训练不如预期
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))

        # critical：梯度回传，准备更新模型参数
        loss.backward()

        # 解决梯度爆炸的问题
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # critical：optimizer更新模型参数
        optimizer.step()

    # 定时打印模型损失，查看模型训练情况
    if (epoch+1) % 5 == 0:
        print("epoch:", epoch, " iter:", i, "loss", loss.item())

    # 定时evaluate模型，查看模型训练情况
    if (epoch+1) % 10 == 0:
        val_loss = evaluate(model, eval_dataloader)

        # critical：根据evaluate的结果，保存最好的模型
        if len(val_losses) == 0 or val_loss < min(val_losses):
            print("epoch:", epoch, "best model, val loss: ", val_loss)
            # critical：使用torch.save()保存模型到路径lm-best.th
            # 之后可以通过torch.load()读取保存好的模型
            torch.save(model.state_dict(), "lm-best.th")

        val_losses.append(val_loss)

