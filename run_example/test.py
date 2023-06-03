import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='maze2d-umaze-v1', )
    parser.add_argument('--e', type=str, default='maze2d--v1')
    return parser.parse_args()

def train(args=get_args()):
    print(args.env)
    print(args.e)
    
if __name__ == '__main__':   
    train()