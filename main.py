import tensorflow as tf

try:
    from data_loaders.data_loader import DataLoader
except ImportError:
    print("Relative import failed")

try:
    from model.model import Model
except ImportError:
    print("Relative import failed")

try:
    from trainers.trainer import Trainer
except ImportError:
    print("Relative import failed")

try:
    from utils.config_parser import process_config_file
except ImportError:
    print("Relative import failed")

try:
    from utils.dirs import create_dirs
except ImportError:
    print("Relative import failed")

try:
    from utlis.logger import Logger
except ImportError:
    print("Relative import failed")

try:
    from utils.utils import get_args
except ImportError:
    print("Relative import failed")

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