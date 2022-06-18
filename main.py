from utils.utils import *
from utils.dataset import DVS_Lip
import time
import numpy as np
from model.model import MSTP
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def test():
    with torch.no_grad():
        dataset = DVS_Lip('test', args)
        logger.info(f'Start Testing, Data Length: {len(dataset)}')
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)        
        
        logger.info('start testing')
        v_acc = []
        label_pred = {i: [] for i in range(args.n_class)}

        for (i_iter, input) in tqdm(enumerate(loader)):
            net.eval()
            tic = time.time()
            event_low = input.get('event_low').cuda(non_blocking=True)
            event_high = input.get('event_high').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            with autocast():
                logit = net(event_low, event_high)

            v_acc.extend((logit.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            label_list = label.cpu().numpy().tolist()
            pred_list = logit.argmax(-1).cpu().numpy().tolist()
            for i in range(len(label_list)):
                label_pred[label_list[i]].append(pred_list[i])

        acc_p1, acc_p2 = compute_each_part_acc(label_pred)
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'test acc: {:.5f}, acc part1: {:.5f}, acc part2: {:.5f}'.format(acc, acc_p1, acc_p2)
        return acc, acc_p1, acc_p2, msg

def train():
    dataset = DVS_Lip('train', args)
    logger.info(f'Start Training, Data Length: {len(dataset)}')
    
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
    loss_fn = nn.CrossEntropyLoss()

    tot_iter = 0
    best_acc, best_acc_p1, best_acc_p2 = 0.0, 0.0, 0.0
    best_epoch = 0
    alpha = 0.2
    scaler = GradScaler()             
    for epoch in range(args.max_epoch):
        for (i_iter, input) in enumerate(loader):
            tic = time.time()

            net.train()
            event_low = input.get('event_low').cuda(non_blocking=True)
            event_high = input.get('event_high').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True).long()

            loss = {}
            with autocast():
                logit = net(event_low, event_high)
                loss_bp = loss_fn(logit, label)

            loss['Total'] = loss_bp
            optimizer.zero_grad()
            scaler.scale(loss_bp).backward()  
            scaler.step(optimizer)
            scaler.update()
            
            toc = time.time()
            if i_iter % 20 == 0:
                msg = 'epoch={},train_iter={},eta={:.5f}'.format(epoch, tot_iter, (toc-tic)*(len(loader)-i_iter)/3600.0)
                for k, v in loss.items():
                    msg += ',{}={:.5f}'.format(k, v)
                msg = msg + str(',lr=' + str(showLR(optimizer)))
                msg = msg + str(',best_acc={:2f}'.format(best_acc))
                logger.info(msg)
            writer.add_scalar('lr', float(showLR(optimizer)), tot_iter)
            writer.add_scalar('loss', loss_bp.item(), tot_iter)
            
            if i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0):
                acc, acc_p1, acc_p2, msg = test()
                logger.info(msg)
                writer.add_scalar('test_acc', acc, tot_iter)
                writer.add_scalar('test_acc/part1', acc_p1, tot_iter)
                writer.add_scalar('test_acc/part2', acc_p2, tot_iter)

                if acc > best_acc:
                    best_acc, best_acc_p1, best_acc_p2, best_epoch = acc, acc_p1, acc_p2, epoch
                    savename = log_dir + '/model_best.pth'
                    temp = os.path.split(savename)[0]
                    if not os.path.exists(temp):
                        os.makedirs(temp)
                    torch.save(net.module.state_dict(), savename)
                    
            tot_iter += 1        
            
        scheduler.step()

    logger.info('best_acc={:2f}'.format(best_acc))
    logger.info('best_acc_part1={:2f}'.format(best_acc_p1))
    logger.info('best_acc_part2={:2f}'.format(best_acc_p2))
    logger.info('best_epoch={:2f}'.format(best_epoch))


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, required=False)
    parser.add_argument('--lr', type=float, required=False, default=3e-4)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--n_class', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--num_workers', type=int, required=False, default=12)
    parser.add_argument('--max_epoch', type=int, required=False, default=80)
    parser.add_argument('--num_bins', type=str2list, required=True, default='1+4')
    parser.add_argument('--test', type=str2bool, required=False, default='false')
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--weights', type=str, required=False, default=None)

    # dataset
    parser.add_argument('--event_root', type=str, default='./data/DVS-Lip')

    # model
    parser.add_argument('--se', type=str2bool, default=False)
    parser.add_argument('--base_channel', type=int, default=64)
    parser.add_argument('--alpha', type=int, default=8)
    parser.add_argument('--beta', type=int, default=5)
    parser.add_argument('--t2s_mul', type=int, default=2)

    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    net = MSTP(args).cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=5e-6)
    logger, writer, log_dir = build_log(args)
    logger.info('Network Arch: ')
    logger.info(net)

    if args.weights is not None:
        logger.info('load weights')
        weight = torch.load(os.path.join('log', args.weights, 'model_best.pth'), map_location=torch.device('cpu'))
        load_missing(net, weight)

    net = nn.DataParallel(net)
    if args.test:
        acc, acc_p1, acc_p2, msg = test()
        logger.info(msg)
        exit()
    train()
