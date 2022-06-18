import time
import os
import argparse
import logging
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_LIST = []

def build_log(args):
    if not args.test:
        if not args.weights:
            timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            log_dir = os.path.join(BASE_DIR, 'log', timestamp)
            if args.log_dir is not None:
                log_dir = os.path.join(BASE_DIR, 'log', args.log_dir)
            DIR_LIST.append(log_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            log_dir = os.path.join(BASE_DIR, 'log', args.weights)
            DIR_LIST.append(log_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'train_log.txt')
        writer = SummaryWriter(os.path.join(log_dir, 'record'))
    else:
        log_file = os.path.join(BASE_DIR, 'log', args.weights, 'test_log.txt')
        writer = None
        log_dir = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    if not args.test:
        s = '-' * 15 + "Start to train" + '-' * 15 + '\n'
        for k, v in args.__dict__.items():
            s += '\t' + k + '\t' + str(v) + '\n'
    else:
        s = '-' * 15 + "Start to test" + '-' * 15 + '\n'
        for k in ['weights']:
            s += '\t' + k + '\t' + str(args.__dict__[k]) + '\n'
    logger.info(s)

    return logger, writer, log_dir

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def str2list(v):
    try:
        return [int(i) for i in v.split('+')]
    except:
        return v.split('+')

def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader =  DataLoader(dataset,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         shuffle = shuffle,
                         drop_last = False,
                         pin_memory=True)
    return loader

def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)

def compute_each_part_acc(label_pred):
    labels = ['accused', 'action', 'allow', 'allowed', 'america', 'american', 'another', 'around', 'attacks', 'banks',
              'become', 'being', 'benefit', 'benefits', 'between', 'billion', 'called', 'capital', 'challenge',
              'change', 'chief', 'couple', 'court', 'death', 'described', 'difference', 'different', 'during',
              'economic', 'education', 'election', 'england', 'evening', 'everything', 'exactly', 'general', 'germany',
              'giving', 'ground', 'happen', 'happened', 'having', 'heavy', 'house', 'hundreds', 'immigration', 'judge',
              'labour', 'leaders', 'legal', 'little', 'london', 'majority', 'meeting', 'military', 'million', 'minutes',
              'missing', 'needs', 'number', 'numbers', 'paying', 'perhaps', 'point', 'potential', 'press', 'price',
              'question', 'really', 'right', 'russia', 'russian', 'saying', 'security', 'several', 'should',
              'significant', 'spend', 'spent', 'started', 'still', 'support', 'syria', 'syrian', 'taken', 'taking',
              'terms', 'these', 'thing', 'think', 'times', 'tomorrow', 'under', 'warning', 'water', 'welcome', 'words',
              'worst', 'years', 'young']
    ambigious_labels = ['action', 'allow', 'allowed', 'america', 'american', 'around', 'being', 'benefit', 'benefits',
                        'billion', 'called', 'challenge', 'change', 'court', 'difference', 'different', 'election',
                        'evening', 'giving', 'ground', 'happen', 'happened', 'having', 'heavy', 'legal', 'little',
                        'meeting', 'million', 'missing', 'needs', 'number', 'numbers', 'paying', 'press', 'price',
                        'russia', 'russian', 'spend', 'spent', 'syria', 'syrian', 'taken', 'taking', 'terms', 'these',
                        'thing', 'think', 'times', 'words', 'worst']
    num_words = len(labels)
    acc = {}
    # pdb.set_trace()
    for i in range(num_words):
        preds = label_pred[i]
        preds_counter = Counter(preds).most_common(num_words)
        acc[labels[i]] = [0.0, len(preds)]
        for j, pred in enumerate(preds_counter):
            if pred[0] == i:
                acc[labels[i]][0] = pred[1]

    acc_part1 = 0
    count_part1 = 0
    acc_part2 = 0
    count_part2 = 0
    for k, v in acc.items():
        if k not in ambigious_labels:
            acc_part1 += v[0]
            count_part1 += v[1]
        else:
            acc_part2 += v[0]
            count_part2 += v[1]

    acc_part1 = acc_part1 / count_part1
    acc_part2 = acc_part2 / count_part2
    # print(count_part1, count_part2)

    return acc_part1, acc_part2

