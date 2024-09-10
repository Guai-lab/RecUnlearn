import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    # TODO 添加配置，优先级：命令行 > 配置文件 > 默认配置

    # 模式选择
    parser.add_argument('--mode', default='train', help='train or test or forget')

    # 训练相关
    parser.add_argument('--dataset', default='ml1m', help='dataset name')
    parser.add_argument('--model', default='mf', help='model name')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr', default=0, type=float, help='learning rate')
    parser.add_argument('--l2_lambda', default=0, type=float, help='l2_lambda')
    parser.add_argument('--seq_len', default=30, type=int, help='seq len')

    # 遗忘相关
    parser.add_argument('--unlearn_ratio', default=0.2, type=float, help='forget rate')
    parser.add_argument('--unlearn_data', default='random', help='unlearn data')
    parser.add_argument('--scale_num', default=100, help='scale num')
    parser.add_argument('--iteration', default=10000, help='iteration')
    # 参数
    parser.add_argument('--cuda', default=0, type=int, help='cuda')

    return vars(parser.parse_args())
