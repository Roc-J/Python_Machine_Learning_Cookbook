# -*- coding:utf-8 -*- 
# Author: Roc-J
# 标记编码方法
from sklearn import preprocessing
# 定义一个标记编码器
label_encoder = preprocessing.LabelEncoder()

# 首先创建一些标记
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

# 现在就可以为这些标记编码
label_encoder.fit(input_classes)
print "\nClass mapping:"
for i, item in enumerate(label_encoder.classes_):
    print item, '-->', i

# 就像前面结果显示的那样，单词被转换成从0开始的索引值。

labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print "\nLabels = ", labels
print "Encoded labels =", list(encoded_labels)

# 这种方式比纯手工进行单词与数字的编码要简单许多。还可以通过数字反转回单词的功能检查结果的正确性

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print "\nEncoded labels = ", encoded_labels
print "Decoded labels = ", list(decoded_labels)
