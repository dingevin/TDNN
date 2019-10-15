
import os
import time
import datetime
import tensorflow as tf
from operator import mul
import numpy as np

import tdnn
import speechloader
import logging
tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):

  batch_size = None
  x1 = tf.placeholder(tf.float32, shape=[batch_size, None, 80])
  y1 = tf.placeholder(tf.int32, shape=[batch_size, None])
  masks = tf.placeholder(tf.float32, shape=[batch_size, None])
  max_seq_len = tf.cast(tf.shape(x1)[1], tf.int32)

  # [-2, 2]
  tdnn1 = tdnn.TDNN(input_dim=80, output_dim=256, input_context=[0,1,2,3,4], sub_sampling=False, batch_norm=True)
  out1 = tdnn1.forward(x1, max_seq_len, 1)

  # {-1, 0, 2} 
  tdnn2 = tdnn.TDNN(input_dim=256, output_dim=256, input_context=[0,1,3], sub_sampling=True, batch_norm=True)
  out2 = tdnn2.forward(out1, max_seq_len-5+1, 2)

  # {-3, 0, 3} 
  tdnn3 = tdnn.TDNN(input_dim=256, output_dim=256, input_context=[0,3,6], sub_sampling=True, batch_norm=True)
  out3 = tdnn3.forward(out2, max_seq_len-5+1-4+1, 3)

  # {-7, 0, 2}
  tdnn4 = tdnn.TDNN(input_dim=256, output_dim=256, input_context=[0,7,9], sub_sampling=True, batch_norm=True)
  out4 = tdnn4.forward(out3, max_seq_len-5+1-4+1-7+1, 4)

  # {0}
  tdnn5 = tdnn.TDNN(input_dim=256, output_dim=256, input_context=[0], sub_sampling=False, batch_norm=True)
  out5 = tdnn5.forward(out4, max_seq_len-5+1-4+1-7+1-10+1, 5)

  # {0}
  tdnn6 = tdnn.TDNN(input_dim=256, output_dim=1629, input_context=[0], sub_sampling=False)
  out6 = tdnn6.forward(out5, max_seq_len-5+1-4+1-7+1-10+1, 6)

  # loss 
  labels = tf.one_hot(y1, 1629, 1.0, 0.0)
  loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out6)
  loss = tf.multiply(loss1, tf.cast(masks, tf.float32))
  loss_mean = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(masks, tf.float32))

  global_step = tf.train.get_or_create_global_step()
  lr = tf.train.exponential_decay(learning_rate=0.0001,
                                  global_step=global_step,
                                  decay_steps=200000,
                                  decay_rate=0.95,
                                  staircase=False)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(lr).minimize(loss_mean)

  # accuracy 
  pred = tf.multiply(tf.nn.softmax(out6), tf.expand_dims(tf.cast(masks, tf.float32), 2))
  log_like = tf.reduce_sum(tf.multiply(pred, labels)) / tf.reduce_sum(tf.cast(masks, tf.float32))
  increment_global_steps = tf.assign_add(global_step, 1)

  # save model
  saver = tf.train.Saver(max_to_keep=100, pad_step_number=True)
  save_path = os.path.join('./model', 'TDNN'+'.ckpt')

  '''
  logger = logging.getLogger()
  while logger.handlers:
    logger.handlers.pop()
  '''

  # check neural network architecture and compute the parameter size
  num_params = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    num_params += reduce(mul, [dim.value for dim in shape], 1)
    tf.logging.info(repr(variable))
  tf.logging.info("{}MB".format(num_params*4/(1024*1024)))
  exit()

  with tf.Session() as sess:
    next_checkpoint_seconds = 0
    save_interval_seconds = 7200
    conf = './conf/train_data_ETC.conf'
    dataloader = speechloader.SpeechLoader(conf)
    sess.run(tf.global_variables_initializer())
    path = tf.train.latest_checkpoint('./model')
    if path:
      tf.logging.info('Load from checkpoint %s.', path)
      saver.restore(sess, path)
      tf.logging.info('Load checkpoint done.')
    while True:
      now = time.time()
      try:
        input, label, mask = dataloader.next()
      except Exception, e:
        continue
      batch_size = input.shape[0]
      _, step, loss, acc, ler = sess.run([train_op, increment_global_steps, loss_mean, log_like, lr],
                                          feed_dict={x1:input, y1:label, masks:mask})

      msg = '%s step:%6d' % (
          datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'), step)
      msg += ' %s:%.8g ' % ('loss', loss)
      msg += ' %s:%.8g ' % ('accuracy', acc)
      msg += ' %s:%.8g ' % ('lr', ler)
      msg += ' %s:%.8g ' % ('batch_size', batch_size)
      tf.logging.info(msg)

      if now > next_checkpoint_seconds:
        tf.logging.info('Save checkpoint')
        path = saver.save(sess, save_path, step)
        tf.logging.info('Save checkpoint done: %s', path)
        next_checkpoint_seconds = now + save_interval_seconds


if __name__ == '__main__':
  tf.app.run(main)

