import tensorflow as tf

class BaseTrainer:
    def __init__(self, sess, model, data, config, logger):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of each epoch :
            - loop over the number of iterations in the config and call the train step
            - add any summaries you want using the summary function
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step :
            - run the tf session
            - return any metrics you need to summarize
        """
        raise NotImplementedError