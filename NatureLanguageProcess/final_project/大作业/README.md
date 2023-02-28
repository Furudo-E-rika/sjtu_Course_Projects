## 运行说明

### Seq2Seq    
在根目录下运行
python scripts/seq2seq_script.py
### Seq2Seq+Attention
在根目录下运行
python scripts/attention_script.py
### Bert
在根目录下运行
python scripts/bert_embed_script.py （bert作为嵌入层）
python scripts/bert_tagging_script.py   （bert作为编码器）
## 代码说明
+ `model/seq2seq.py`: Seq2Seq模型的第一个版本实现
+ `model/seq2seq2.py`: 将编码器、解码器、结果输出层单独抽象出来的Seq2Seq模型
+ `model/attention.py`: 各种版本的注意力类
+ `model/bert_embedding.py`: 使用bert预训练模型作为嵌入层
+ `model/bert_tagging.py`: 使用bert预训练模型作为编码器
+ `model/pointer.py`: 指针网络实现（缺少数据集读取方式的改写）
+ `scripts/seq2seq_script.py`: Seq2Seq模型的运行代码
+ `scripts/attention_script.py`: Attention模型的运行代码
+ `scripts/bert_embed_script.py`:: 使用bert预训练模型作为嵌入层的模型运行代码
+ `scripts/bert_tagging_script.py`:: 使用bert预训练模型作为编码器的模型运行代码
