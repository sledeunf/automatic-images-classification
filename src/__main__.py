import argparse, sys

# disable tensorflow warnings before imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from .database import Database

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # parse arguments to create dabase
    create_db_parser = subparsers.add_parser('create-database')
    create_db_parser.set_defaults(action='create-database')
    create_db_parser.add_argument('--classes', nargs='*', 
        help='(optional) specify classes to select in source folder')
    create_db_parser.add_argument('--from', type=str, required=True, help='path to the source folder (here coreldb)')

    # parse arguments to train CNN
    train_cnn_parser = subparsers.add_parser('train-cnn')
    train_cnn_parser.set_defaults(action='train-cnn')
    train_cnn_parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    train_cnn_parser.add_argument('-e', '--epochs', type=int, default=15, help='epochs')
    train_cnn_parser.add_argument('--history', action='store_true', help='plot training history')

    # parse arguments to classify image using trained CNN
    cnn_classify_parser = subparsers.add_parser('cnn-classify')
    cnn_classify_parser.set_defaults(action='cnn-classify')
    cnn_classify_parser.add_argument('--confusion', action='store_true', help='plot confusion matrix after testing')

    #
    args = parser.parse_args()
    if not getattr(args, 'action', None):
        parser.print_help()
        sys.exit(2)

    return args

DATABASE_NAME = 'database'

def main():
    args = parse_args()

    if args.action == 'create-database':
        Database.create(DATABASE_NAME, getattr(args, 'from'), classes=args.classes)
    
    elif args.action == 'train-cnn':
        from .convolutional_nn import CNNClassifier
        database = Database(DATABASE_NAME)
        model = CNNClassifier(len(database))
        model.train(database, batch_size=args.batch_size, 
                epochs=args.epochs, history=args.history, overwrite=True)

    elif args.action == 'cnn-classify':
        from .convolutional_nn import CNNClassifier
        database = Database(DATABASE_NAME)
        model = CNNClassifier(len(database))
        # this step is skipped if model exists
        model.train(database, overwrite=False)
        model.classify_test_images(database, confusion_matrix=args.confusion)
        


if __name__ == '__main__':
    main()
