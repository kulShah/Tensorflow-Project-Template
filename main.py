# import tensorflow as tf

from data_loaders.data_loader import DataLoader
from model.model import Model
from trainers.trainer import Trainer
from utils.config_parser import process_config_file
from utils.dirs import create_dirs
from utlis.logger import Logger
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the JSON config file
    try:
        args = get_args()
        config = process_config_file(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiment dirs
    create_dirs([config.summary_dir, config.checkpoints_dir])
    # create tf session
    sess = tf.Session()
    # create data loader
    data = DataLoader(config)

    # create an instance of the model
    model = Model(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the components to it
    trainer = Trainer(sess, model, data, logger, config)
    # load model if it exists
    model.load(sess)
    # train the model
    trainer.train()

if __name__ == '__main__':
    main()