该作业项目以jittor深度学习框架完成了CIFAR-10数据集的分类问题。

执行根目录下的Train.py文件能够完成完整CIFAR数据集的加载与训练，
相应地，执行Train_10percent.py文件能够完成切分后的数据集的加载训练以及后续模型优化后的训练过程。
数据集接口文件见根目录下的Dataloader.py。
其中数据集文件保存在./cifar-10-batched-py目录下。
CNN和Resnet模型代码保存在./models目录下。
训练后的模型文件保存在./trained_models目录下。
训练过程中的训练误差以及测试准确率曲线保存在./output目录下。

相关报告可见根目录下的DL_assign2_report.pdf。
