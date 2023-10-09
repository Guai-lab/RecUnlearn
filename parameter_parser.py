import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    # 模式选择
    parser.add_argument('--mode', default='train', choices=['train', 'test', 'forget'], help='train or test or forget')

    # 训练相关
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--model', default='ConvNet', help='model name')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=1, type=int, help='epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    # 遗忘相关
    parser.add_argument('--is_forget', default='False', help='whether to forget')

    # 参数
    # parser.add_argument('--cuda', default=0, type=int, help='cuda number')

    return vars(parser.parse_args())
