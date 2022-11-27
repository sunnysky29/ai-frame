# -*- coding: utf-8 -*-
"""
==============================================================================
Time : 2022/11/26 22:41
File : pytorch 训练代码 GPU 版本demo.py

# File              : gcn_imdb_sentence_classification.py
# Author            : admin <admin>
# Date              : 31.12.2021
# Last Modified Date: 06.01.2022
# Last Modified By  : admin <admin>

运行(2卡为例)： python -m torch.distributed.launch --nproc_per_node=n_gpus xxx.py
             python -m torch.distributed.launch --nproc_per_node=2 xxx.py

- 1)https://blog.csdn.net/scar2016/article/details/124404318
- 2) [哔站：33、完整讲解PyTorch多GPU分布式训练代码编写](https://www.bilibili.com/video/BV1xZ4y1S7dG/?spm_id_from=pageDriver&vd_source=abeb4ad4122e4eff23d97059cf088ab4)
- 3) [关于Pytorch中的Embedding padding](http://www.linzehui.me/2018/08/19/%E7%A2%8E%E7%89%87%E7%9F%A5%E8%AF%86/%E5%85%B3%E4%BA%8EPytorch%E4%B8%ADEmbedding%E7%9A%84padding/)
- 4） IMDB 数据集链接：http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
==============================================================================
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.datasets import IMDB
# pip install torchtext 安装指令
from torchtext.datasets.imdb import NUM_LINES
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

import sys
import os
import logging
# 配置日志的输出形式
logging.basicConfig(
    # 设置logging输出的级别为logging.WARN在控制台中打印出来
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

VOCAB_SIZE = 15000
# 第一期： 编写GCNN模型代码：门卷积网络模型
class GCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, num_class=2):
        """

        :param vocab_size: 单词表的大小，根据 datasets 进行统计
        :param embedding_dim: 每一个token我们用一个向量来表示，向量长度表示为embedding_dim
        :param num_class: 分类的种类个数
        """
        # 对父类进行初始化，这样就可以调用父类nn.Module里面的相关函数
        super(GCNN, self).__init__()

        # 创建一个nn.Embedding的词表，
        # 行是单词表的个数vocab_size,列表示每个单词向量的长度
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        # 用xavier_uniform_来初始化嵌入表的权重值
        nn.init.xavier_uniform_(self.embedding_table.weight)

        # 设置一维卷积
        self.conv_A_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7)

        self.conv_A_2 = nn.Conv1d(64, 64, 15, stride=7)
        self.conv_B_2 = nn.Conv1d(64, 64, 15, stride=7)

        # 定义全连接层
        self.output_linear1 = nn.Linear(64, 128)
        # 定义分类全连接层，num_class为分类数
        self.output_linear2 = nn.Linear(128, num_class)

    def forward(self, word_index):
        # 定义GCN网络的算子操作流程，基于句子单词ID输入得到分类logits输出

        # 1. 通过word_index得到word_embedding
        # word_index shape:[bs, max_seq_len]
        # nn.Embedding(vocab_size, embedding_dim)
        # [bs,max_seq_len,embedding_dim]
        word_embedding = self.embedding_table(word_index) #[bs, max_seq_len, embedding_dim]

        # 2. 编写第一层1D门卷积模块
        # [bs,max_seq_len,embedding_dim] -> [bs,embedding_dim,max_seq_len]
        word_embedding = word_embedding.transpose(1, 2) #[bs, embedding_dim, max_seq_len]
        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)
        H = A * torch.sigmoid(B) #[bs, 64, max_seq_len]

        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B) #[bs, 64, max_seq_len]

        # 3. 池化并经过全连接层
        pool_output = torch.mean(H, dim=-1) #平均池化，得到[bs, 64]
        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output) #[bs, 2]

        return logits


class TextClassificationModel(nn.Module):
    """ 简单版embeddingbag+DNN模型 """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, num_class=2):
        super(TextClassificationModel, self).__init__()
        # self.embedding.shape = [bs,embedding_dim]
        # nn.EmbeddingBag词袋是在行与行之间进行求均值
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        # 再将得到的均值用全连接层映射到分类数上

        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, token_index):
        embedded = self.embedding(token_index) # shape: [bs, embedding_dim]
        return self.fc(embedded)


# step2 构建IMDB DataLoader
BATCH_SIZE = 64 * 2 # 2张卡
data_path = r'/Users/dufy/code/corpus/txt_classification'

# dataset里面的数据分词化
def yield_tokens(train_data_iter, tokenizer):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        # 将 comment即x进行分词化处理
        yield tokenizer(comment)

# 得到dataset类型对象
train_data_iter = IMDB(root=data_path, split='train') # Dataset类型的对象

# 实例化一个分词器
tokenizer = get_tokenizer("basic_english")

# 将频次小于20的词用"<unk>"代替，将频次高于20的词装入到单词表中Vocab
vocab = build_vocab_from_iterator(yield_tokens(train_data_iter, tokenizer),
                                  min_freq=20, specials=["<unk>"])

# 将不在单词表里面的数据索引值设置为0
vocab.set_default_index(0)
print(f"单词表大小: {len(vocab)}")

def collate_fn(batch):
    """ 对DataLoader所生成的mini-batch进行后处理 """
    target = []
    token_index = []
    max_length = 0

    # batch接受的也是一个元祖
    # label = [bs,y];
    # comment = [bs,x]
    for i, (label, comment) in enumerate(batch):
        tokens = tokenizer(comment)

        token_index.append(vocab(tokens))
        if len(tokens) > max_length:
            max_length = len(tokens)

        if label == "pos":
            target.append(0)
        else:
            target.append(1)

    token_index = [index + [0]*(max_length-len(index)) for index in token_index]
    return (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32))


# step3 编写训练代码
def train(local_rank, train_data_set, eval_data_set, model, optimizer, num_epoch, log_step_interval, save_step_interval, eval_step_interval, save_path, resume=""):
    """ 此处data_loader是map-style dataset """
    start_epoch = 0
    start_step = 0
    if resume != "":
        #  加载之前训过的模型的参数文件
        logging.warning(f"loading from {resume}")
        # ==================================================
        # 可以是  cpu, cuda, cuda:index
        # checkpoint = torch.load(resume, map_location=torch.device("cpu"))
        checkpoint = torch.load(resume, map_location=torch.device("cuda:0"))
        # ==================================================

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        print(f'接着从step {start_step} 开始训练')
# ==================================================
    model = nn.parallel.DistributedDataParallel(model.cuda(local_rank),
                                                device_ids=[local_rank])  # 模型拷贝, 放入 DistributedDataParallel()
    train_sampler = DistributedSampler(train_data_set)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=BATCH_SIZE,
                                                    collate_fn=collate_fn, sampler=train_sampler)
    eval_data_loader = torch.utils.data.DataLoader(eval_data_set, batch_size=8,
                                                    collate_fn=collate_fn)

# ==================================================

    for epoch_index in range(start_epoch, num_epoch):
        ema_loss = 0.
        num_batches = len(train_data_loader)

        # ==================================================
        train_sampler.set_epoch(epoch_index)  # 为了让每张卡在每个周期中得到的数据是随机的
        # ==================================================

        for batch_index, (target, token_index) in enumerate(train_data_loader):
            optimizer.zero_grad()
            step = num_batches*(epoch_index) + batch_index + 1
            # 数据拷贝==================================================
            # TORCH.TENSOR.CUDA()
            # Returns a copy of this object in CUDA memory
            token_index = token_index.cuda(local_rank)
            target = target.cuda(local_rank)
            # ==================================================
            logits = model(token_index)
            bce_loss = F.binary_cross_entropy(torch.sigmoid(logits), F.one_hot(target, num_classes=2).to(torch.float32))
            ema_loss = 0.9*ema_loss + 0.1*bce_loss
            bce_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if step % log_step_interval == 0:
                logging.warning(f"epoch_index: {epoch_index}, batch_index: {batch_index}, ema_loss: {ema_loss.item()}")

            if step % save_step_interval == 0 and local_rank== 0:
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, f"step_{step}.pt")
                torch.save({
                    'epoch': epoch_index,
                    'step': step,
                    # 'model_state_dict': model.state_dict(),
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': bce_loss,
                }, save_file)
                logging.warning(f"checkpoint has been saved in {save_file}")

            if step % eval_step_interval == 0:
                logging.warning("start to do evaluation...")
                model.eval()
                ema_eval_loss = 0
                total_acc_account = 0
                total_account = 0
                for eval_batch_index, (eval_target, eval_token_index) in enumerate(eval_data_loader):
                    total_account += eval_target.shape[0]
                    eval_logits = model(eval_token_index)
                    total_acc_account += (torch.argmax(eval_logits, dim=-1) == eval_target).sum().item()
                    eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits), F.one_hot(eval_target, num_classes=2).to(torch.float32))
                    ema_eval_loss = 0.9*ema_eval_loss + 0.1*eval_bce_loss
                acc = total_acc_account/total_account

                logging.warning(f"eval_ema_loss: {ema_eval_loss.item()}, eval_acc: {acc}")
                model.train()
        print(f'-----------step: {step}')


# step4 测试代码
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        help='local device id on current node')
    args = parser.parse_args()

    # 单机多卡
    if torch.cuda.is_available():
        logging.warning(f'Cuda is available')
        if torch.cuda.device_count() > 1:
            logging.warning(f'Find {torch.cuda.device_count()} gpus!')
        else:
            logging.warning('Too few gpus!!')
    else:
        logging.warning(f'Cuda is not available ! Exit')

    n_gpus =2
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)

    # model = GCNN()
    model = TextClassificationModel()
    print("模型总参数:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data_iter = IMDB(root=data_path, split='train') # Dataset类型的对象
    train_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(train_data_iter), batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    eval_data_iter = IMDB(root=data_path, split='test') # Dataset类型的对象
    eval_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(eval_data_iter), batch_size=8, collate_fn=collate_fn)
    resume = ""
    # resume = r"../data/logs_imdb_text_classification/step_500.pt"  # 加载 check_point

    train(args.local_rank, to_map_style_dataset(train_data_iter), to_map_style_dataset(eval_data_iter),
          model, optimizer,
          num_epoch=10, log_step_interval=20, save_step_interval=500,
          eval_step_interval=300, save_path="../data/logs_imdb_text_classification", resume=resume)

