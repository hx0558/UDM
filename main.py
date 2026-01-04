import os
import time
import argparse
import datetime
from pathlib import Path
import numpy as np
from datasets import *
from engine import evaluate, train_one_epoch, get_flops
from util import (create_logger, save_checkpoint,
                  get_dataloaders, set_seed, resume_ckpt,
                  get_mixup_fn, get_transform,
                  build_optimizer, build_scheduler, build_criterion)
from network import *
from model import UDM

def parse_args():
    parser = argparse.ArgumentParser('UDM', add_help=False)
    parser.add_argument('--lr', default=(1e-4), type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--ckpt_interval', default=250, type=int)
    parser.add_argument('--clip_max_norm', default=0, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--seed', default=0, type=int,
                        help='base seed for experiments (seed + 1~5)')
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--scheduler', default='step', type=str)
    parser.add_argument('--flops', action='store_true')

    parser.add_argument('--dataset_root', default='D:/UDM/datasets/Houston',type=str)
    parser.add_argument('--dataset',default='Houston', type=str)
    parser.add_argument('--output_dir', default='logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--exp_name', default='multi_cls', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--mode', default=0, type=int, help='HSI or 1==LiDAR')
    # training tricks
    parser.add_argument('--mixup', default=False, type=bool,
                        help='mixup and cutmix data argumentation')
    parser.add_argument('--smooth', default=0., type=float,
                        help='label smoothing')

    # 梯度调制
    parser.add_argument("--modulation_starts", type=int, default=1)
    parser.add_argument("--modulation_ends", type=int, default=300)
    parser.add_argument("--zeta", type=float, default=1)

    # model parameters
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--hsi_channel', type=int, default=224)
    parser.add_argument('--lidar_channel', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=27)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--dim', type=int, default=256)         # Houton MUUFL 256 Berlin 128
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--hidden_c', type=int, default=128)
    parser.add_argument('--hidden_s', type=int, default=128)
    parser.add_argument('--is_cls_token', type=bool, default=False)
    parser.add_argument('--in_channels', type=int, default=128)
    parser.add_argument('--mlp_head', type=str, default='None')

    parser.add_argument('--ablation', type=int, default=1)

    parser.add_argument('--fusionloss', type=float, default=0)

    args, unparsed = parser.parse_known_args()
    return args




def main(args, logger, seed, exp_time, acc, A, k,ECE, triT, tesT):
    # prepare seed
    set_seed(seed)

    # build dataset and dataloader
    builder = CASILiDARBiloder(args.dataset_root, seed=seed)
    data_sets = builder.get_datasets()
    data_loaders = get_dataloaders(data_sets, args, 4)
    num_classes = builder.get_num_classes()

    logger.info(f'Dataset builder info: \n{builder}')
    num_bands_HSI = builder.get_num_bands()[0]
    num_bands_LiDAR = builder.get_num_bands()[1]
    args.hsi_channel = num_bands_HSI
    args.lidar_channel = num_bands_LiDAR
    # training tricks
    mixup_fn = get_mixup_fn(args.smooth, num_classes) if args.mixup else None
    if args.mode == 0:
        num_bands_HSI = num_bands_HSI
    else:
        num_bands_HSI = num_bands_LiDAR

    model = UDM(args)

    model = model.cuda()
    print(model)

    # calculate params and flops
    total_parameters = sum(p.numel() for p in model.parameters())
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    logger.info(f'total params: {total_parameters} | train params: {n_parameters}')

    flops = get_flops(model, data_sets.train, logger)

    # build optimizer, scheduler and criterion
    optimizer = build_optimizer(args.optimizer, model, lr=args.lr)
    logger.info(f'{optimizer.__class__} is loaded as the optimizer.')
    lr_scheduler = build_scheduler(args.scheduler, optimizer, len(data_loaders.train),
                                   epochs=args.epochs, lr=args.lr, decay_rate=0.1,decay_epochs=args.lr_drop)
    logger.info(f'{lr_scheduler.__class__} is loaded as the lr_scheduler.')
    criterion = build_criterion(mixup=args.mixup, smooth=args.smooth)
    logger.info(f'{criterion.__class__} is loaded as the criterion.')

    best_acc = 0.0
    # auto_resume, resume and pretrain  # TODO
    if args.resume:
        resume_ckpt(model, optimizer, lr_scheduler, args)

    # ###################### TRAIN_VAL ###################### #
    print("Start training")

    train_start=time.time()
    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            args, model, data_loaders.train, criterion, optimizer, lr_scheduler, mixup_fn,
            args.clip_max_norm, epoch, args.epochs, logger)

        val_acc = evaluate(args, seed, model, data_loaders.val, criterion, acc, A, k, ECE, exp_time, logger=logger)[0] \
            if epoch % 2 == 1 and (epoch + 1) > (args.epochs - 1) else 0.

        save_checkpoint(args.output_dir, args.ckpt_interval, epoch, model, optimizer,
                        lr_scheduler, logger, best_acc, val_acc, args.lr_drop)
        best_acc = max(best_acc, val_acc)
    train_end=time.time()

    print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
    # ###################### Evaluate ###################### #
    test_acc, _, acc, A, k, ECE = evaluate(args, seed, model, data_loaders.test, criterion, acc, A, k, ECE, exp_time, logger=logger)
    save_checkpoint(model=model, save_path=Path(args.output_dir) / f'last_model_{test_acc:.4f}.pth')
    logger.info(f'Last model accuracy: {test_acc:.4f}')

    model.load_state_dict(torch.load(Path(args.output_dir) / 'best_ckpt.pth')['model'])
    test_acc, _, acc, A, k, ECE = evaluate(args, seed, model, data_loaders.test, criterion, acc, A, k, ECE, exp_time, logger=logger)
    test_end = time.time()

    print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
    save_checkpoint(model=model, save_path=Path(args.output_dir) / f'best_model_{test_acc:.4f}.pth')
    logger.info(f'Best model accuracy: {test_acc:.4f}')
    triT[exp_time - 1] = train_end - train_start
    tesT[exp_time - 1] = test_end - train_end
    return test_acc, acc, A, k,ECE, total_parameters, flops, triT, tesT


def command(args):
    # args = parse_args()
    builder = CASILiDARBiloder(args.dataset_root, seed=12)
    args.num_classes = builder.num_classes
    args.hsi_channel = builder.num_bands[0]
    args.lidar_channel = builder.num_bands[1]
    print(args)
    nClass = args.num_classes
    # prepare directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    time_tag1 = time.strftime("%Y-%m-%d_%H-%M-%S")
    # logger
    logger = create_logger(output_dir=args.output_dir, name=(str(args.lr)+'_'+time_tag1))
    logger.info(args)

    start_time = time.time()

    # experiments with seeds of args.seed+1~5
    acc_list = []
    EXP_TIMES = 5
    acc = np.zeros([EXP_TIMES, 1])
    A = np.zeros([EXP_TIMES, nClass])
    k = np.zeros([EXP_TIMES, 1])
    ECE = np.zeros([EXP_TIMES, 1])
    TRtime = np.zeros([EXP_TIMES, 1])
    TEtime = np.zeros([EXP_TIMES, 1])

    for seed in range(len(acc_list) + 1, EXP_TIMES + 1):
        logger.info(f'EXP {seed}: SEED = {args.seed + seed}')
        test_acc, acc, A, k, ECE, para, flops, TRtime, TEtime = main(args, logger, args.seed + seed, seed, acc, A, k,
                                                                     ECE, TRtime, TEtime)

    AA = np.mean(A, 1)
    AAMean = np.mean(AA, 0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(acc)
    OAStd = np.std(acc)
    kMean = np.mean(k)
    kStd = np.std(k)
    ECEMean = np.mean(ECE)
    ECEStd = np.std(ECE)

    TRtimeMean = np.mean(TRtime)
    TRtimeStd = np.std(TRtime)
    TEtimeMean = np.mean(TEtime)
    TEtimeStd = np.std(TEtime)

    print("average OA: " + "{:.2f}".format(OAMean) + " ± " + "{:.2f}".format(OAStd))
    print("average AA: " + "{:.2f}".format(100 * AAMean) + " ± " + "{:.2f}".format(100 * AAStd))
    print("average kappa: " + "{:.4f}".format(100 * kMean) + " ± " + "{:.4f}".format(100 * kStd))
    print("average ECE: " + "{:.4f}".format(100 * ECEMean) + " ± " + "{:.4f}".format(100 * ECEStd))
    for i in range(nClass):
        print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + "±" + "{:.2f}".format(100 * AStd[i]))

    print(f"average Train Time:{TRtimeMean:.3f} ± {TRtimeStd:.3f}")
    print(f"average Test Time:{TEtimeMean:.3f} ± {TEtimeStd:.3f}")
    print(f"Parameters:{para}")
    print(f"Flops:{flops}")
    total_time = time.time() - start_time
    print(f'total time: {total_time}')
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Run time: {}'.format(total_time_str))

    time_tag = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{args.epochs}_{args.dim}_{args.lr}_{args.dataset}_results_{time_tag}.txt"

    fold_path = "D:/UDM/ablation"
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
        print(f"{fold_path} has been created")
    else:
        print(f"{fold_path} is already existed")

    with open(f"{fold_path}" + "/" + f"{save_path}", 'w') as f:
        f.write(f"average OA: {OAMean:.2f} ± {OAStd:.2f}\n")
        f.write(f"average AA: {100 * AAMean:.2f} ± {100 * AAStd:.2f}\n")
        f.write(f"average kappa: {100 * kMean:.4f} ± {100 * kStd:.4f}\n")
        f.write(f"average ECE: {100 * ECEMean:.4f} ± {100 * ECEStd:.4f}\n")

        for i in range(nClass):
            f.write(f"Class {i}: {100 * AMean[i]:.2f} ± {100 * AStd[i]:.2f}\n")

        f.write(f"average Train Time:{TRtimeMean:.3f} ± {TRtimeStd:.3f}\n")
        f.write(f"average Test Time:{TEtimeMean:.3f} ± {TEtimeStd:.3f}\n")
        f.write(f"Parameters:{para}\n")
        f.write(f"Flops:{flops}\n")

    print(f"Results saved to: {save_path}")

if __name__ == '__main__':
    args = parse_args()
    command(args)