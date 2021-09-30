import h5py
import numpy as np
import argparse
import sys, os
import subprocess
import csv, yaml
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import horovod.torch as hvd
except:
    hvd = None

## Investigate server
nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
print("NTHREADS=", nthreads, "CPU_COUNT=", os.cpu_count())
torch.set_num_threads(nthreads)

## Make argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', action='store', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=128, help='Batch size')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--lr', action='store', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batchPerStep', action='store', type=int, default=1, help='Number of batches per step (to emulate all-reduce)')
parser.add_argument('--shuffle', action='store', type=bool, default=True, help='Shuffle batches for each epochs')
parser.add_argument('--optimizer', action='store', choices=('sgd', 'adam', 'radam', 'ranger',
    'lamb', 'novograd'), default='adam', help='optimizer to run')
parser.add_argument('--model', action='store', choices=('default', 'defaultnorm1', 'defaultnorm0', 'defaultcat', 'defaultnorm1cat',
                                                        'log3ch', 'log3chnorm1', 'log3chnorm0', 'log3chcat', 'log3chnorm1cat',
                                                        'log5ch', 'log5chnorm1', 'log5chnorm0', 'log5chcat', 'log5chnorm1cat',
                                                        'original',
                                                        'circpad', 'circpadnorm1', 'circpadnorm0', 'circpadnorm1cat',
                                                        'circpadlog3ch', 'circpadlog3chnorm1', 'circpadlog3chnorm0', 'circpadlog3chnorm1cat',
                                                        'circpadlog5ch', 'circpadlog5chnorm1', 'circpadlog5chnorm0', 'circpadlog5chnorm1cat',
                                                        'CPlargekernel', 'CPlargekernelnorm0', 'CPlargekernelnorm1',),
                               default='default', help='choice of model')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('-c', '--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('--lars','-l', action='store', type=bool, default=False, help='Use LARS optimizer')
parser.add_argument('--noperfmon', action='store_true', default=False, help='turn off performance monitoring')

args = parser.parse_args()

## Load config file to read train data info
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

## horovod setup
hvd_rank, hvd_size = 0, 1
if hvd:
    hvd.init()
    hvd_rank = hvd.rank()
    hvd_size = hvd.size()
    print("Horovod is available. (rank=%d size=%d)" % (hvd_rank, hvd_size))
    if torch.cuda.is_available(): torch.cuda.set_device(hvd.local_rank())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

## Setup result file I/O directories
if not os.path.exists(args.outdir): os.makedirs(args.outdir)
modelFile = os.path.join(args.outdir, 'model.pth')
weightFile = os.path.join(args.outdir, 'weight_%d.pth' % hvd_rank)
predFile = os.path.join(args.outdir, 'predict_%d.npy' % hvd_rank)
trainingFile = os.path.join(args.outdir, 'history_%d.csv' % hvd_rank)
resourceByCPFile = os.path.join(args.outdir, 'resourceByCP_%d.csv' % hvd_rank)
resourceByTimeFile = os.path.join(args.outdir, 'resourceByTime_%d.csv' % hvd_rank)

if not args.noperfmon:
    proc = subprocess.Popen(['python', '../scripts/monitor_proc.py', '-t', '1',
                            '-o', resourceByTimeFile, '%d' % os.getpid()],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

import time
class TimeHistory():
    def on_train_begin(self):
        self.times = []
    def on_epoch_begin(self):
        self.epoch_time_start = time.time()
    def on_epoch_end(self):
        self.times.append(time.time() - self.epoch_time_start)

sys.path.append("../scripts")
from monitor_proc import SysStat
sysstat = SysStat(os.getpid(), fileName=resourceByCPFile)
sysstat.update(annotation="start_loggin")

## Start loading data
sys.path.append("../python")
from HEPCNN.dataset_hepcnn import HEPCNNDataset as MyDataset

sysstat.update(annotation="add samples")
print("Start loading samples")
myDataset = MyDataset()
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    myDataset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    myDataset.setProcessLabel(name, sampleInfo['label'])
sysstat.update(annotation="init dataset")
print("Start initializing dataset")
myDataset.initialize(logger=sysstat)

sysstat.update(annotation="split dataset")
lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())

kwargs = {'num_workers':min(config['training']['nDataLoaders'], nthreads), 'pin_memory':False}

if hvd:
    trnSampler = torch.utils.data.distributed.DistributedSampler(trnDataset, num_replicas=hvd_size, rank=hvd_rank)
    valSampler = torch.utils.data.distributed.DistributedSampler(valDataset, num_replicas=hvd_size, rank=hvd_rank)
    trnLoader = DataLoader(trnDataset, batch_size=args.batch, sampler=trnSampler, **kwargs)
    valLoader = DataLoader(valDataset, batch_size=args.batch, sampler=valSampler, **kwargs)
else:
    trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
    valLoader = DataLoader(valDataset, batch_size=args.batch, shuffle=False, **kwargs)

## Build model
sysstat.update(annotation="Model start")
if args.model == 'original':
    from HEPCNN.model_original import MyModel
elif 'largekernel' in args.model:
    from HEPCNN.model_largekernel import MyModel
elif 'circpad' in args.model:
    from HEPCNN.model_circpad import MyModel
else:
    from HEPCNN.model_default import MyModel
model = MyModel(myDataset.width, myDataset.height, model=args.model)

## Decide which device to use (cpu or gpu)
if hvd_rank == 0: torch.save(model, modelFile)
device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

# Decide optimizer
if args.optimizer == 'radam':
    from optimizers.RAdam import RAdam
    optm = RAdam(model.parameters(), lr=args.lr)
elif args.optimizer == 'ranger':
    from optimizers.RAdam import RAdam
    from optimizers.Lookahead import Lookahead
    optm_base = RAdam(model.parameters(), lr=args.lr)
    optm = Lookahead(optm_base)
elif args.optimizer == 'adam':
    optm = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optm = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'lamb':
    from optimizers.lamb import Lamb
    optm = Lamb(model.parameters(), lr=args.lr)
elif args.optimizer == 'novograd':
    from optimizers.novograd import NovoGrad
    optm = NovoGrad(model.parameters(), lr=args.lr)
else:
    print("Cannot find optimizer in the list")
    exit()

# Decide whether to use LARS or not (LARS not working with horovod)
if args.lars == True:
    from optimizers.lars import LARS
    optm_base = optm
    optm = LARS(optimizer=optm_base)
    print("Using LARS with base optimizer %s"%(args.optimizer))

## Additional horovod setup
if hvd:
    compression = hvd.Compression.none
    optm = hvd.DistributedOptimizer(optm,
                                    named_parameters=model.named_parameters(),
                                    compression=compression, backward_passes_per_step=args.batch)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optm, root_rank=0)

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

## Finishing train setup
sysstat.update(annotation="modelsetup_done")

with open(args.outdir+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()

## Start training
from tqdm import tqdm
from sklearn.metrics import accuracy_score
bestWeight, bestLoss = {}, 1e9
try:
    timeHistory = TimeHistory()
    timeHistory.on_train_begin()
    sysstat.update(annotation="train_start")
    history = {'time':[], 'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
    ## Start each epoch
    for epoch in range(args.epoch):
        timeHistory.on_epoch_begin()
        sysstat.update(annotation='epoch_begin')
        ## Start train step
        model.train()
        trn_loss, trn_acc = 0., 0.
        optm.zero_grad()

        for i, (data, label, weight, rescale, procIdx, evtWeight) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, args.epoch))):
            data = data.float().to(device)
            label = label.float().to(device)
            rescale = rescale.float().to(device)
            weight = weight.float().to(device)*rescale
            evtWeight = evtWeight.float().to(device)*weight

            pred = model(data)
            crit = torch.nn.BCEWithLogitsLoss(weight=evtWeight)
            if device == 'cuda': crit = crit.cuda()
            l = crit(pred.view(-1), label)
            l.backward()
            if i % args.batchPerStep == 0 or i+1 == len(trnLoader):
                optm.step()
                optm.zero_grad()

            trn_loss += l.item()
            trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight=weight.to('cpu'))

            sysstat.update()
        trn_loss /= len(trnLoader)
        trn_acc  /= len(trnLoader)
        ## Start validation step
        model.eval()
        val_loss, val_acc = 0., 0.
        for i, (data, label, weight, rescale, procIdx, evtWeight) in enumerate(tqdm(valLoader)):
            data = data.float().to(device)
            label = label.float().to(device)
            rescale = rescale.float().to(device)
            weight = weight.float().to(device)*rescale
            evtWeight = evtWeight.float().to(device)*weight

            pred = model(data)
            crit = torch.nn.BCEWithLogitsLoss(weight=evtWeight)
            loss = crit(pred.view(-1), label)

            val_loss += loss.item()
            val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight=weight.to('cpu'))
        val_loss /= len(valLoader)
        val_acc  /= len(valLoader)

        if hvd: val_acc = metric_average(val_acc, 'avg_accuracy')
        if bestLoss > val_loss:
            bestWeight = model.to('cpu').state_dict()
            bestLoss = val_loss
            if hvd_rank == 0:
                torch.save(bestWeight, weightFile)
                sysstat.update(annotation="saved_model")
            model.to(device)

        timeHistory.on_epoch_end()
        sysstat.update(annotation='epoch_end')
        history['loss'].append(trn_loss)
        history['acc'].append(trn_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        history['time'].append(timeHistory.times[-1])
        if hvd_rank == 0:
            with open(trainingFile, 'w') as f:
                writer = csv.writer(f)
                keys = history.keys()
                writer.writerow(keys)
                for row in zip(*[history[key] for key in keys]):
                    writer.writerow(row)
            sysstat.update(annotation="wrote_logs")

    sysstat.update(annotation="train_end")

except KeyboardInterrupt:
    print("Training finished early")
