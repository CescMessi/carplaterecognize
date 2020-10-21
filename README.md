# carplaterecognize
使用pytorch和opencv的简易识别车牌程序

使用opencv找到类似车牌的物体，接着使用模型判断是否为车牌。若为车牌，将车牌图像拉伸至标准形状，对字符进行分割，每个字符单独使用模型进行识别。

模型训练代码已经包含，使用的是简单的squeezenet，可以自行修改。
