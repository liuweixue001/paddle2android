paddle模型从训练到生成可部署的android模型

--以mnist数据集为 例

LeNet.py -> 定义模型

train.py -> 训练模型

predict.py -> 预测模型

export_model.py -> 导出推理模型(需要注意LeNet的前向传播函数需要转为动态图)

inference2android.py -> 推理模型转为安卓模型

inferenceandroid.txt -> Ai studio下推理模型转为安卓模型的命令

makepic.py -> 生成[1, 28, 28]图像

a.jpg -> 黑色单通道图像

model -> paddle模型

inference -> 推理模型

android_model ->安卓模型

others:

android demo地址：

https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo

使用自己的model、label、image替换image_classification_demo\app\src\main\assets下的对应文件

最后修改image_classification_demo\app\src\main\res\values\string.xml下的：

name="MODEL_PATH_DEFAULT">models/模型文件夹名

name="LABEL_PATH_DEFAULT">labels/标签.txt

name="IMAGE_PATH_DEFAULT">images/初始化图片.jpg<

注意：

由于demo为三通道图片，需要训练三通道模型才能在正常运行app，否则会显示模型载入失败

可以修改官方demo的其它内容以适配单通道模型，因为单通道图片很少用，暂没研究

