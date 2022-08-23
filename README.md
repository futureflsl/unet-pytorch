# unet-pytorch
源码来自：https://download.csdn.net/download/MissLemonh/85048033 

讲解：https://blog.csdn.net/MissLemonh/article/details/122373594 


使用：将labelme标注的数据集直接转化为mask，然后分别将图片和mask放到对应目录，注意labelme转化的数据集需要自己写脚本去 

把图片和mask名字对应起来。然后修改train.py参数训练即可。我已经测试过8类训练和预测效果还行，具体还需要自己学习代码和修改。 
