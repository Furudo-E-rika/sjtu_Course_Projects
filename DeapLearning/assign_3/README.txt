该作业项目以jittor深度学习框架完成了CIFAR-10数据集下的图片拼接问题。

执行根目录下的Train.py文件能够完成拼图模型的训练和测试过程，
执行根目录下的Pretrained_Classification.py文件能够完成以拼图模型作为预训练方式的分类模型训练与测试过程。

其中数据集文件保存在./cifar-10-batched-py目录下，
CIFAR-10数据集的分类接口和拼图接口保存在./Dataloader目录下。
拼图模型DeepPermNet和分类模型CNN的代码保存在./models目录下。
训练后的模型文件保存在./trained_models目录下。
训练过程中的训练误差以及测试准确率曲线保存在./output目录下。

相关报告及报告中的插图可见./report目录。
