import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        default='/basket/ABIDE_Data/data/RAW/NYU/h5_2c_win5',
        type=str,
        help='Directory path of the data')
    parser.add_argument(
        '--MAT_dir',
        default= '/MAT',
        type = str,
        help = 'Directory path of subjects ID'
    )
    parser.add_argument(
        '--csv_dir',
        default= '//basket/ABIDE_Data/data/RAW/NYU/MAT/NYU.csv',
        type = str,
        help = 'Directory path of subjects ID'
    )
    parser.add_argument(
        '--architecture',
        default= 'NC3D',
        type = str,
        help = 'Choose network architecture from NC3D and ResNet and CRNN'
    )
    parser.add_argument(
        '--cross_val',
        default = True,
        help = 'If true, do cross validation'
    )
    parser.add_argument(
        '--fold',
        default = 0,
        type = int,
        help = 'If no cross validation, you need to spevify a certain split to '
    )
    parser.add_argument(
        '--result_path',
        default='/results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help=
        'Number of classes'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=2,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--sample_size',
        default=64,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--win_size',
        default=2,
        type=int,
        help='Num of input channels')
    parser.add_argument(
        '--s_sz',
        default=5,
        type=int,
        help='Sliding window size for calculation avg and std channel/ lstm time length')
    parser.add_argument(
        '--rep',
        default=5,
        type=int,
        help='repeat times when open a subject')
    parser.add_argument(
        '--sample_duration',
        default=175,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--weights',
        default=[176/105,176/71],
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-2, type=float, help='Weight Decay')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=True)
    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--nb_filter',
        default= [16,32,64,128],
        type=list,
        help='Number of kernels of each layer.'
    )
    parser.add_argument(
        '--batch_size', default=300, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default= 100,
        type= int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--n_val_samples',
        default=300,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train= False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val= False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test= True)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--use_cuda', action='store_true', help='If true, cuda is used.')
    parser.set_defaults(use_cuda = True)
    parser.add_argument(
        '--n_threads',
        default=10,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=5,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args