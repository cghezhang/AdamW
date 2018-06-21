"""
3 STEPs:
1. Initialize Optimizer Controller ap = AdamWParameter(...)
2. Create Optimizer AdamOptimizer(...)
3. For each training batch, run ap.getParameter(epoch) to get current learning_rate and weightDecay rate
"""

from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer


#STEP 1
number_example     = 1000      #How many training samples in one epoch
learning_rate      = 0.0001
weightDecay        = 0.0025
batch_size         = 32
training_epochs    = 150
Te                 = 10
Tmult              = 2
total_batch        = int(number_example / batch_size)

ap = AdamWParameter(nEpochs      = training_epochs, 
                    Te           = Te,
                    Tmult        = Tmult,
                    LR           = learning_rate, 
                    weightDecay  = weightDecay,
                    batchSize    = batch_size,
                    nBatches     = total_batch
                    )



#STEP 2
lr_c = tf.placeholder(tf.float32)
wd_c = tf.placeholder(tf.float32)
optimizer = AdamOptimizer(learning_rate=lr_c, wdc =wd_c).minimize(loss)



#STEP 3
"""During Training"""
'''
for i in range(num_trainBatch):
        batch_xs = ...
        batch_ys = ...
        clr, wdc = ap.getParameter(epoch)   #currentLearningRate & weightDecayCurrent
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, lr_c: clr, wd_c: wdc}
        l, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
'''
