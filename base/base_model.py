import tensorflow as tf

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.init_global_step() # init the global step
        self.init_cur_epoch() # init the epoch counter

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("saving model....")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("model saved")

    # load the latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("model loaded")

    # initialize a tf variable to use it as an epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # initialize a tf variable to use it as a global step counter
    def init_global_step(self):
        # Add the global step tensor to the tf trainer
        with tf.Variable('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='gloabl_step')

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError