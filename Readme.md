# 介绍
本文档介绍公选课DCS2021:《自然语言处理绪论》的期末大作业

## 任务介绍
根据示例代码，实现一个简单的英文RNN语言模型，包括模型实现，训练和预测模块。

## 评估标准
1. 模型在训练集和验证集的损失
2. 给定缺少最后一个词语的句子，预测缺失词语（给出概率分布）

## 任务要求
1. 不允许作弊抄袭，不能完全照抄示例代码
2. Deadline：2022-12-31

# 文件介绍
model.py: 示例代码，实现了中文RNN语言模型的部分功能
train/eval.txt: 中文RNN语言模型的训练/验证集
train/eval/test_en.txt: 英文RNN语言模型的训练/验证/测试集

# 依赖包的安装
运行model.py需要自行安装jieba和**pytorch**，jieba之前已经使用过。而pytorch是一个非常常用又相对简单的深度学习python包，在自己实现语言模型时也可以使用pytorch。下面简单介绍它的安装

## 最简单的方法
pip install pytorch
到pytorch官网的[下载页面](https://pytorch.org/get-started/previous-versions/)有更详细的介绍

## 常见问题
下载超时：因为pip默认服务器在国外，百度搜索pip换源，改为国内的阿里云节点下载即可
下载好了却用不了：尝试pip3 install pytorch
提示和已有环境冲突：pytorch有很多个版本，pip会一直试，总有一款适合你。想彻底解决请使用[Anaconda](https://www.anaconda.com/)。

## GPU版本pytorch安装
一般来说使用最简单的方法安装pytorch即可，但深度学习一般使用GPU，而pip默认安装CPU版pytorch。GPU版pytorch和CPU版的差别在于几分钟和几小时完成训练。想安装GPU版pytorch，需要其他包和显卡驱动的配合，具体请参考pytorch官网的[下载页面](https://pytorch.org/get-started/previous-versions/)和百度。

# 简单提示
## 关于任务
实现一个语言模型，或者一个机器学习/深度学习的模型，大致可以划分为以下几部分：
1. 文本预处理：怎么分词？标点符号怎么处理？大小写怎么处理？怎么把字词转化为one-hot向量或者数字标记？怎么讲数据集划分成batch
2. 模型实现：怎么把处理好的文本tensor输入，要依次经过那些模块，tensor中途会发生怎样的形状上的变化
3. 训练验证部分：怎么计算损失？怎么更新模型参数？怎么评估模型和保存最佳checkpoint
4. 预测部分：怎么把模型的输入转化为原来的文字

一份简单的深度学习模型的完整代码，基本由上面几部分组成。虽说简单，但代码量也并不少，按上述的划分去阅读示例代码，有助于系统了解代码每个部分的功能。

## 关于示例代码
1. 请仔细阅读注释，标有**critical**为重要部分
2. 遇到不了解的torch函数请自行查阅[pytorch官方文档](https://pytorch.org/docs/stable/index.html)
3. 遇到bug却不知道哪里错了？使用pdb包进行debug，通过插入import pdb; pdb.set_trace()在需要的地方（比如bug之前）插入断点，然后逐步追凶，更多pdb包的使用请百度。使用vsocde和pycharm软件自带的debug工具更方便。