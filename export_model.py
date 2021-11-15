# 导入Paddle相关包
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.nn import Layer
from LeNet import LeNet


model = LeNet(num_classes=10)
try:
    model.load_dict(paddle.load("./model/mnist.pdparams"))
    print("model has loaded")
except:
    raise ("no model was found")
# 保存预测格式模型
paddle.jit.save(model, './inference/mnist')