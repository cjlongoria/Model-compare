import argparse

def get_args():
    parser = argparse.ArgumentParser('standard argument parser')

    parser.add_argument('--lr',
                        type=int,
                        default=.2,
                        help='Learning rate for the model')
    parser.add_argument('--decay',
                        type=int,
                        default=0.0001,
                        help='decay rate per epoch')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs to train on')

    args = parser.parse_args()

    return args