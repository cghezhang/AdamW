import numpy as np
import tensorflow as tf
import PFNNParameter as PFNN
from PFNNParameter import PFNNParameter
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
import os.path

tf.set_random_seed(23456)  

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

H0 = X_nn[:,:-1] 
H0 = tf.expand_dims(H0, -1)       
H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

b0 = tf.expand_dims(P0.bias, -1)      
H1 = tf.matmul(P0.weight, H0) + b0      
H1 = tf.nn.elu(H1)             
H1 = tf.nn.dropout(H1, keep_prob=keep_prob) 

b1 = tf.expand_dims(P1.bias, -1)       
H2 = tf.matmul(P1.weight, H1) + b1       
H2 = tf.nn.elu(H2)                
H2 = tf.nn.dropout(H2, keep_prob=keep_prob) 

b2 = tf.expand_dims(P2.bias, -1)       
H3 = tf.matmul(P2.weight, H2) + b2      
H3 = tf.squeeze(H3, -1)          


loss = tf.reduce_mean(tf.square(Y_nn - H3))


learning_rate      = 0.0001
learning_rateDecay = 0.0025

'''
learning_rate      = 0.001
learning_rateDecay = 0.025
'''
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


#batch size and epoch
batch_size = 32
training_epochs = 200
total_batch = int(number_example / batch_size)
print("totoal_batch:", total_batch)

#randomly select training set
I = np.arange(number_example)
rng.shuffle(I)


#training set and  test set
num_testBatch  = np.int(total_batch/10)
num_trainBatch = total_batch - num_testBatch
print("training_batch:", num_trainBatch)
print("test_batch:", num_testBatch)

   
#used for saving errorof each epoch
error_train = np.ones(training_epochs)
error_test  = np.ones(training_epochs)

#start to train
print('Learning start..')
for epoch in range(training_epochs):
    avg_cost_train = 0
    avg_cost_test  = 0

    '''  
    #modify parameters in adamw
    clr, wdc = ap.getParameter(epoch)   #currentLearningRate & weightDecayCurrent
    optimizer._lr    = clr
    optimizer._lr_t  = clr 
    optimizer._wdc   = wdc
    optimizer._wdc_t = wdc
    '''
    
    for i in range(num_trainBatch):
        index_train = I[i*batch_size:(i+1)*batch_size]
        batch_xs = input_x[index_train]
        batch_ys = input_y[index_train]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7}
        l, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost_train += l / num_trainBatch
        
        #modify parameters in adamw
        clr, wdc = ap.getParameter(epoch)   #currentLearningRate & weightDecayCurrent
        optimizer._lr    = clr
        optimizer._lr_t  = clr 
        optimizer._wdc   = wdc
        optimizer._wdc_t = wdc
        if i % 1000 == 0:
            print(i, "trainingloss:", l)
            
    for i in range(num_testBatch):
        if i==0:
            index_test = I[-(i+1)*batch_size: ]
        else:
            index_test = I[-(i+1)*batch_size: -i*batch_size]
        batch_xs = input_x[index_test]
        batch_ys = input_y[index_test]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 1}
        testError = sess.run(loss, feed_dict=feed_dict)
        avg_cost_test += testError / num_testBatch
        if i % 1000 == 0:
            print(i, "testloss:",testError)
    
    #print and save training test error 
    print('Epoch:', '%04d' % (epoch + 1), 'trainingloss =', '{:.9f}'.format(avg_cost_train))
    print('Epoch:', '%04d' % (epoch + 1), 'testloss =', '{:.9f}'.format(avg_cost_test))
    print('Epoch:', '%04d' % (epoch + 1), 'clr:', sess.run(optimizer._lr), sess.run(optimizer._lr_t))
    print('Epoch:', '%04d' % (epoch + 1), 'wdc:', sess.run(optimizer._wdc),sess.run(optimizer._wdc_t))
    error_train[epoch] = avg_cost_train
    error_test[epoch]  = avg_cost_test
    error_train.tofile("./dog/model/error_train.bin")
    error_test.tofile("./dog/model/error_test.bin")
    
    #save model and weights
    save_path = saver.save(sess, "./dog/model/model.ckpt")
    PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                      (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                      50, 
                      './dog/nn'
                      )    
    
    #save weights every 5epoch
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
