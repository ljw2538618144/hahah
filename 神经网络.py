# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:04:02 2019

@author: Mechrevo
"""

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random 

# 构造满足一元二次方程的函数
# 构建300个点，区间[-1, 1]，直接用numpy生成等差数列，然后将结果是300个点的一维数组转换为300X1二维数组
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 此处加入噪声点，使得他们与x_data的维度一致，并且拟合成均值为0，方差为0.05正态分布
noise = np.random.normal(0, 0.05, x_data.shape)
# 假设的方程，并且加入噪声点
y_data = np.square(x_data) + random.random()-25 +noise
#占位符，None表示任意尺寸
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 定义网络层
def definate_layer(inputs, in_size, out_size, activation_function=None):
    # 权重构造，in_size*out_size大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 偏置构造，1*out_size大小的矩阵,值为0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 做乘法，即x*w+b，当然也可以转置权重矩阵，写为w*x+b
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    # 输出数据
    return outputs
# 构建两个隐藏层，每一层都有20个神经单元
h1 = definate_layer(xs, 1, 20, activation_function=tf.nn.relu)
h2 = definate_layer(h1, 20, 20, activation_function=tf.nn.relu)
# 构建输出层，与输入层一样，包含一个神经元
prediction = definate_layer(h2, 20, 1, activation_function=None)
#打印三层的值
print('h1:'+str(h1))
print('h2:'+str(h2))
print('prediction:'+str(prediction))
 
# 构建损失函数，计算预测值与真实值的误差
#reduction_indices：默认值是None，即把input_tensor降到 0维，也就是一个数。
#对于2维input_tensor，reduction_indices=0时，按列；reduction_indices=1时，按行。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=1))
#使用Adam优化算法、学习率为0.001来最小化损失
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
# 初始化，构建TensorFlow会话
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 训练100000次
for i in range(10000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    prediction_value=sess.run(prediction,feed_dict={xs:x_data})
    if i%100 == 0: # 100次打印一次
        print("loss: ", sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
print('prediction_value:'+str(prediction_value)) #显示预测值
print(prediction_value.shape)  #显示预测值的形状  
#显示x_data和y_data的内容和形状
print('x_data:'+str(x_data))
print('y_data:'+str(y_data))
print('x_data.shape:'+str(x_data.shape))
print('y_data.shape:'+str(y_data.shape))  
#可视化拟合结果        
fig=plt.figure()
bx = fig.add_subplot(1,1,1)
bx.scatter(x_data,y_data)
#bx.scatter(x_data,prediction_value)#用scatter画散点图
bx.plot(x_data, prediction_value,'r-',lw=1)#用plot画线图
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.show()