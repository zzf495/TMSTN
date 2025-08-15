import argparse

def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/path_of_dataset/Office31/')
    parser.add_argument('--path_of_source_model', type=str, help='The path of source model',
                        default='/checkpoints/Office31/source/best.pkl')
    parser.add_argument('--dataset', type=str, help='Path to save results',
                        default='Office31')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='amazon')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='webcam')
    parser.add_argument('--num_classes', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--use_cuda', help="Use GPU?", type=str, default='cuda')
    parser.add_argument('--backbone', type=str,
                        help='The backbone of model', default='vit')
    parser.add_argument('--dimension', type=int,
                        help='Dimension of latent layers', default=512)
    parser.add_argument('--alpha', type=float,
                        help='Weight of MSML', default=1.0)
    parser.add_argument('--beta', type=float,
                        help='Weight of TSPE', default=1.0)
    parser.add_argument('--gamma', type=float,
                        help='Weight of IM', default=1.0)
    parser.add_argument('--delta', type=float,
                        help='Weight of CE', default=1.0)
    parser.add_argument('--smooth', type=float, help='value of label smoothing', default=1e-5)  # 0.005
    parser.add_argument('--gar', type=float, help='number of micro-community = (1/gar)+1', default=0.2)  # 0.005
    parser.add_argument('--LDS_type', type=str, default="full", help="Full training or part")
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=40)
    parser.add_argument('--warm_up_iter', type=int,
                        help='Number of warm up ', default=10)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.008)
    parser.add_argument('--lr_decay', type=float, help='lr_decay', default=0.75)
    parser.add_argument('--lr_gamma', type=float, help='lr_gamma', default=0.001)
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=15)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2025)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args