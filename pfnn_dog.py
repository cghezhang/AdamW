import numpy as np
import tensorflow as tf
import PFNNParameter as PFNN
from PFNNParameter import PFNNParameter
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
import os.path


tf.set_random_seed(23456)  
'''
#load data
data = np.float32(np.loadtxt('Data.txt'))


#A,B,C stores continuous 3 frames in same animation sequence 
A = []
B = []
C = []
for i in range(len(data)-2):
    if(data[i,0] == data[i+1,0] and data[i,0] == data[i+2,0]):
        A.append(data[i])
        B.append(data[i+1])
        C.append(data[i+2])
A = np.asarray(A)
B = np.asarray(B)
C = np.asarray(C)
number_example =len(A)    #number of training data


num_joint=21              #number of joint
num_trajectory = 12       #number of points in trajectory
num_style = 8             #number of style
offset= 3                 #first 3 item in data are no use.
jointNeurons = 6*num_joint  #pos, vel, trans 
trajectoryNeurons = (8+num_style)*num_trajectory #pos, dir,hei, style
    
#input 
X = np.concatenate(
        (
                B[:,offset+jointNeurons:offset+jointNeurons+trajectoryNeurons], #trajectory pos, dir,hei, style of B
                A[:,offset:offset+jointNeurons]                                 #joint pos, vel, trans rot vel magnitudes of A
        ),axis = 1) 


#get trajecoty positionX,Z velocityX,Z of future trajectory for Y
Traj_out = np.float32(np.zeros((number_example,np.int(num_trajectory/2*4))))
Traj_out_start = np.int(offset+ jointNeurons+ num_trajectory/2*6)
for i in range(np.int(num_trajectory/2)):
    Traj_out[:,i*4:(i+1)*4] = C[:,[Traj_out_start,Traj_out_start+2,Traj_out_start+3,Traj_out_start+5]]
    Traj_out_start += 6

#output    
Y = np.concatenate(
        (
                Traj_out, 
                B[:,offset:offset+jointNeurons], 
                B[:,offset+jointNeurons+trajectoryNeurons+1:]
        ),axis = 1)

P = B[:,offset+jointNeurons+trajectoryNeurons]
P = P[:,np.newaxis]


#normalize data
Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

for i in range(Xstd.size):
    if (Xstd[i]==0):
        Xstd[i]=1
for i in range(Ystd.size):
    if (Ystd[i]==0):
        Ystd[i]=1

X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd

#save mean and std
Xmean.tofile('./dog/data/Xmean.bin')
Ymean.tofile('./dog/data/Ymean.bin')
Xstd.tofile('./dog/data/Xstd.bin')
Ystd.tofile('./dog/data/Ystd.bin')


input_x = np.concatenate((X,P),axis = 1) #input of nn, including X and P
input_y = Y

input_size  = input_x.shape[1]
output_size = input_y.shape[1]


print("Data is processed")
'''



""" Phase Function Neural Network """
"""input of nn"""
X_nn = tf.placeholder(tf.float32, [None, input_size], name='x-input')
Y_nn = tf.placeholder(tf.float32, [None, output_size], name='y-input')


"""parameter of nn"""
rng = np.random.RandomState(23456)
nslices = 4                             # number of control points in phase function
phase = X_nn[:,-1]                      #phase
P0 = PFNNParameter((nslices, 512, input_size-1), rng, phase, 'wb0')
P1 = PFNNParameter((nslices, 512, 512), rng, phase, 'wb1')
P2 = PFNNParameter((nslices, output_size, 512), rng, phase, 'wb2')



keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
#structure of nn
H0 = X_nn[:,:-1] #input of nn     dims:  ?*342
H0 = tf.expand_dims(H0, -1)       #dims: ?*342*1
H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

b0 = tf.expand_dims(P0.bias, -1)      #dims:  ?*512*1
H1 = tf.matmul(P0.weight, H0) + b0      #dims:  ?*512*342 mul ?*342*1 = ?*512*1
H1 = tf.nn.elu(H1)               #get 1th hidden layer with 'ELU' funciton
H1 = tf.nn.dropout(H1, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

b1 = tf.expand_dims(P1.bias, -1)       #dims: ?*512*1
H2 = tf.matmul(P1.weight, H1) + b1       #dims: ?*512*512 mul ?*512*1 = ?*512*1
H2 = tf.nn.elu(H2)                #get 2th hidden layer with 'ELU' funciton
H2 = tf.nn.dropout(H2, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

b2 = tf.expand_dims(P2.bias, -1)       #dims: ?*311*1
H3 = tf.matmul(P2.weight, H2) + b2       #dims: ?*311*512 mul ?*512*1 =?*311*1
H3 = tf.squeeze(H3, -1)           #dims: ?*311



#this might be the regularization that Dan use
def regularization_penalty(a0, a1, a2, gamma):
    return gamma * (tf.reduce_mean(tf.abs(a0))+tf.reduce_mean(tf.abs(a1))+tf.reduce_mean(tf.abs(a2)))/3

cost = tf.reduce_mean(tf.square(Y_nn - H3))
loss = cost + regularization_penalty(P0.alpha, P1.alpha, P1.alpha, 0.01)


#
learning_rate      = 0.0001
learning_rateDecay = 0.0025
batch_size         = 32
training_epochs    = 200
Te                 = 20
Tmult              = 2 
total_batch        = int(number_example / batch_size)

ap = AdamWParameter(nEpochs      = training_epochs, 
                    Te           = Te,
                    Tmult        = Tmult,
                    LR           = learning_rate, 
                    weightDecay  = learning_rateDecay,
                    batchSize    = batch_size,
                    nBatches     = total_batch
                    )
clr, wdc = ap.getParameter(0)
optimizer = AdamOptimizer(learning_rate=clr, wdc =wdc ).minimize(loss)


#session

sess = tf.Session()
sess.run(tf.global_variables_initializer())

"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
sess.run(tf.global_variables_initializer()) 
"""

#saver for saving the variables
saver = tf.train.Saver()



#start to train
print('Learning start..')
print("totoal_batch:", total_batch)
I = np.arange(number_example)
rng.shuffle(I)
error = np.ones(training_epochs)
for epoch in range(training_epochs):
    avg_cost = 0
    
    clr, wdc = ap.getParameter(epoch)
    optimizer._lr    = clr
    optimizer._lr_t  = clr 
    #optimizer._wdc_t  = wdc 
    

    for i in range(total_batch):
        index_train = I[i*batch_size:(i+1)*batch_size]
        batch_xs = input_x[index_train]
        batch_ys = input_y[index_train]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7}
        l,c, _, = sess.run([loss,cost, optimizer], feed_dict=feed_dict)
        avg_cost += l / total_batch
        
        if i % 1000 == 0:
            print(i, "loss:", l, "cost:", c)
    
    save_path = saver.save(sess, "./dog/model/model.ckpt")
    PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                      (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                      50, 
                      './dog/nn'
                      )
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_cost))
    error[epoch] = avg_cost
    error.tofile("./dog/model/error.bin")
    
    if epoch>0 and epoch%5==0:
        path_dog  = './dog/weights/dogNN%03i' % epoch
        if not os.path.exists(path_dog):
            os.makedirs(path_dog)
        PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                          (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                          50, 
                          path_dog)
    
      
print('Learning Finished!')
#-----------------------------above is model training----------------------------------
