数据说明：
* weak: 每个音频片段标注了有哪些事件发生
* unlabel_in_domain: 无标注，事件分布与 `weak` 的事件分布相似
* unlabel_out_of_domain: 无标注，事件分布与 `weak` 的事件分布无关

准备好环境后, 跑 baseline 步骤 (注意将环境初始化部分改成自己的设置):

1. 提取特征：
```bash
cd data;
sbatch prepare_data.sh /dssg/home/acct-stu/stu464/data/domestic_sound_events
cd ..;
```

2. 训练、测试:
```bash
sbatch run.sh
```

注: evaluate.py 用于计算指标，预测结果 `prediction.csv` 写成这样的形式 (分隔符为 `\t`):
```
filename        event_label     onset   offset
Y09RRavdW3C0_30.000_40.000.wav  Speech  0.000   1.000
YIZ_zfkNcxRQ_61.000_71.000.wav  Blender 8.000   9.000
......
```
调用方法：
```bash
python evaluate.py --prediction prediction.csv \
                   --label data/eval/label.csv \
                   --output result.txt
```

3. 项目结构

基本模型结构存储于 /models/baseline.py中
各数据增强方式存储于 /augmentation.py中
实验结果（除训练好的模型）存储于/experiments中
模型及数据的相关配置存储于/configs中


#### 若想复现最佳结果，请在SJTU HPC中输入以下指令：
```bash
sbatch run.sh
```
