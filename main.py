import cv2, os
import math
import matplotlib.pyplot as plt
import random
import imutils
import numpy as np
import tensorflow as tf
import time
from vgg_model import vgg

start = time.time() # 시작 시간 설정
print(vgg.model())
x_data = tf.placeholder(shape = (None,224,224,3), dtype=np.float32) # 4차원, datatype
# placeholder -> 공간을 만드는 것. None-> batch_size가 뭐가 들어가도 상관없어 진다
y_data = tf.placeholder(shape = (None,3), dtype = np.float32)
training_bool = tf.placeholder(dtype=bool)

output = vgg(x_data,training_bool)
prob = tf.nn.softmax(output)    # 확률화 시키는 것
output_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_data, logits = prob)  # 정답과 output을 비교해 loss를 끌어냄
loss = tf.reduce_mean(output_loss)  # 10 개의 loss의 평균을 내는 것(batch size만큼의 개수를 평균을 내는 것)

# update_ops는 공식같은 것(통상적으로 쓰는 것)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opti = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)  # loss의 평균값을 최소화한다

init = tf.global_variables_initializer()    # 임의의 값이 들어있는 것을 초기화 시키는 것


path = 'C:/Users/bmn'

skin_cancer_dx = ['bkl','mel','nv']
total_name_list = []    # 모든 이미지의 이름을 뜻함
y_list = [] # 이미지의 labeling(bkl인지 mel인지 nv인지)

for one_class in skin_cancer_dx:
    one_class_path = path + '/' + one_class
    for one_img in os.listdir(one_class_path):  # 해당 path내에 있는 파일명을 가져온다
        total_name_list.append(one_class_path + '/' + one_img)
        y_list.append(one_class)

total_name_list = total_name_list[:1200]
y_list = y_list[:1200]

tmp = [[x,y] for x,y in zip(total_name_list, y_list)]
random.shuffle(tmp)
total_name_list = [n[0] for n in tmp]
y_list = [n[1] for n in tmp]

for i in range(len(y_list)):    # y_list에 들어있는 것들을 숫자로 바꿔준다
    y_list[i] = skin_cancer_dx.index(y_list[i])

split_cnt = int(len(total_name_list) * 0.8) # train set와 validation set를 나눔
training_img_set = total_name_list[:split_cnt]
val_img_set = total_name_list[split_cnt:]

training_y_set = y_list[:split_cnt]
val_y_set = y_list[split_cnt:]

best_accuracy = 0   # 가장 높은 정확도
accuracy = []   # 정확도 들의 모음

epoch = 100
batch_size = 20  # batch_size : 데이터를 분할해서 학습하는 크기
batch_cnt = int(math.ceil(len(training_img_set)/batch_size))
val_batch_cnt = int(math.ceil(len(val_img_set)/batch_size))

train_img_loss = []
train_img_accuracy = []
val_img_loss = []
val_img_accuracy = []

with tf.Session() as sess:  # 합성곱 연산을 하겠다는 것을 설정하는 것
    sess.run(init)
    saver = tf.train.Saver()    # 모델을 불러오고 저장하는 것
    for k in range(epoch):  # 전체 이미지 개수에 대해서 epoch만큼 반복함
        train_total_loss = 0
        train_total_accuracy = 0
        val_total_accuracy = 0
        val_total_loss = 0
        for i in range(batch_cnt):
            one_batch = training_img_set[batch_size *i:batch_size*i + batch_size]  # batch size만큼씩 자르는 것
            one_batch_y = training_y_set[batch_size *i:batch_size*i + batch_size]   # batch size만큼씩 자르는 것
            one_batch_y = np.eye(len(skin_cancer_dx))[one_batch_y]  # 해당 인덱스를 1로 만들고, 그 외 값을 0으로 만든다
            train_img_set = []

            for one_img in one_batch:
                img = cv2.imread(one_img)
                flip_int = random.randint(0,3)

                if flip_int == 0:
                    pass
                elif flip_int == 1:
                    img = cv2.flip(img,0)
                elif flip_int == 2:
                    img = cv2.flip(img,1)
                elif flip_int == 3: # 상하 및 좌우 반전
                    img = cv2.flip(img,0)
                    img = cv2.flip(img,1)

                #print(img.shape)    # 이미지 크기
                x = int(img.shape[1] * 0.2)
                y = int(img.shape[0] * 0.2)
                x_shift_int = random.randint(-x,x)  # 이미지 크기의 20%이내로 이동하는 것
                y_shift_int = random.randint(-y,y)

                img = imutils.translate(img,x_shift_int,y_shift_int)    # 좌표축 이동
                rotation_int = random.randint(-30, 30)
                img = imutils.rotate(img, rotation_int)  # 회전

                img = cv2.resize(img,(224, 224))
                train_img_set.append(img)

            train_img_set = np.array(train_img_set, dtype=np.float32)

            _, batch_prob, batch_loss = sess.run([opti, prob, loss], feed_dict = {x_data : train_img_set, y_data : one_batch_y, training_bool : True})

            train_accuracy = np.mean(np.cast[np.int32](np.equal(np.argmax(batch_prob,axis = 1), np.argmax(one_batch_y,axis = 1))))
            # argmax: 가장 높은 값을 가진 곳의 위치를 뽑는 것. axis의 순서(2번째) 가 세로축을 의미
            # equal: 둘이 같은지 확인하여 bool형으로 뽑음. cast: bool형 변수를 정수로 바꿔준다

            train_total_accuracy += train_accuracy  # 한 배치마다의 평균 정확도를 계속 더하는 것
            train_total_loss += batch_loss
           # 평균을 내주는 것
        train_total_loss /= batch_cnt
        train_total_loss = round(train_total_loss,4)
        train_total_accuracy /= batch_cnt
        train_total_accuracy = round(train_total_accuracy,2)
        train_img_loss.append(train_total_loss)
        train_img_accuracy.append(train_total_accuracy)

        one_epoch = k+1
        print(one_epoch,'epoch')
        print('train_loss:', train_total_loss)
        print('train_accuracy:', train_total_accuracy)


        for i in range(val_batch_cnt):  # 20%, 정확도를 내기 위한 것. test img set이나 마찬가지다
            one_batch_1 = val_img_set[batch_size *i:batch_size*i + batch_size]
            one_batch_y_1 = val_y_set[batch_size *i:batch_size*i + batch_size]
            one_batch_y_1 = np.eye(len(skin_cancer_dx))[one_batch_y_1]

            one_batch_img_set = []
            for one_img in one_batch_1:
                img = cv2.imread(one_img)
                img = cv2.resize(img,(224,224))
                one_batch_img_set.append(img)
            one_batch_img_set = np.array(one_batch_img_set, dtype=np.float32)
            batch_prob_1, batch_loss_1 = sess.run([prob, loss], feed_dict = {x_data : one_batch_img_set, y_data : one_batch_y_1, training_bool : False})
            val_accuracy = np.mean(np.cast[np.int32](np.equal(np.argmax(batch_prob_1,axis = 1), np.argmax(one_batch_y_1,axis = 1))))
            val_total_accuracy += val_accuracy
            val_total_loss += batch_loss_1

        val_total_accuracy /= val_batch_cnt
        val_total_accuracy = round(val_total_accuracy,2)
        val_total_loss /= val_batch_cnt
        val_total_loss = round(val_total_loss,4)
        val_img_loss.append(val_total_loss)
        val_img_accuracy.append(val_total_accuracy)

        onepoch = k+1
        print(onepoch, 'epoch')
        print('val_loss:', val_total_loss)
        print('val_accuracy:',val_total_accuracy)

print("time_spent: ", time.time() - start)

# 그래프 출력
epoch_num = []
for i in range(epoch):
    epoch_num.append(i+1)
plt.plot(epoch_num,train_img_loss)
plt.plot(epoch_num,val_img_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Experiement result:loss')
plt.legend(['train','validation'])
plt.show()

plt.plot(epoch_num,train_img_accuracy)
plt.plot(epoch_num,val_img_accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Experiement result:accuracy')
plt.legend(['train','validation'])
plt.show()

# top-5 accuracy 출력
val_img_accuracy.sort()
val_img_accuracy.reverse()
for top_5 in range(5):
    top_5_num = top_5+1
    print('top',top_5_num,'val_accuracy:', val_img_accuracy[top_5])


