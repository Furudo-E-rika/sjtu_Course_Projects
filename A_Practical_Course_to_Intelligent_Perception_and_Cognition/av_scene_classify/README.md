# 跑 baseline 配置

baseline 用到 openl3 提取的 audio 和 visual 特征，此步骤过程较长，已预先提取好放于 `/lustre/home/acct-stu/stu168/ai3611/av_scene_classify/data/feature`，基于此跑 baseline (注意将环境初始化部分改成自己的设置):


注: eval_pred.py 用于计算指标，预测结果 `prediction.csv` 写成这样的形式 (分隔符为 `\t`):
```
aid     scene_pred      airport     bus     ......  tram
airport-lisbon-1175-44106   airport     0.9   0.000   ......  0.001
......
```
# 项目运行
首先，我们需要在根目录下运行以下代码来获取音视频特征数据：
```bash 
bash extract_openl3_emb.sh
```
接着，根据想要训练和评估的模型，我们可以运行不同的.sh文件。如想复现效果最好的早期融合模型，请运行以下指令：
```bash
sbatch run_early.sh
```


# 项目结构
该项目的目录结构如下：
```
- av_scene_classify\
    - configs\ # 配置文件 （模型的具体定义见报告）
        - attention.yaml # 注意力模型配置文件
        - attention2.yaml # 注意力模型2配置文件
        - audio_only.yaml # 只使用音频特征的模型配置文件
        - video_only.yaml # 只使用视频特征的模型配置文件
        - early_fusion.yaml # 早期融合模型配置文件
        - late_fusion.yaml # 晚期融合模型配置文件
        - hybrid_fusion.yaml # 混合融合模型配置文件
        
    - data\ # 数据集
        ...
    - experiments\ # 训练结果 （注：由于训练好的模型文件较大，因此没有上传这些文件）
        ...
    - models\ # 模型定义
        - attention.py # 注意力模型
        - attention2.py # 注意力模型2
        - audio_only.py # 只使用音频特征的模型
        - video_only.py # 只使用视频特征的模型
        - early_fusion.py # 早期融合模型
        - late_fusion.py # 晚期融合模型
        - hybrid_fusion.py # 混合融合模型
        
    - slurm_logs\ # 使用交大超算时训练日志的存储位置
        ...
        
 ```
