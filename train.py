
import __init__
import os

#os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda-11.1/bin64:/usr/local/cuda-11.2/bin64' 

import numpy as np
import torch
import torch.multiprocessing as mp
import torch_geometric.datasets as GeoData
from torch_geometric.loader import DenseDataLoader
import torch_geometric.transforms as T
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from config import OptInit
from architecture import DenseDeepGCN, CustomDenseDeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
import logging
from tqdm import tqdm
from parallel_wrapper import launch
import comm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='log/mlp4')

def train(model, train_loader, optimizer, criterion, opt, cur_rank):
    opt.losses.reset()
    model.train()
    with tqdm(train_loader) as tqdm_loader:
        for i, data in enumerate(tqdm_loader):
            opt.iter += 1
            desc = 'Epoch:{}  Iter:{}  [{}/{}]  Loss:{Losses.avg: .4f}'\
                .format(opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses)
            tqdm_loader.set_description(desc)

            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            gt = data.y.to(opt.device)
            # ------------------ zero, output, loss
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, gt)

            # ------------------ optimization
            loss.backward()
            optimizer.step()

            opt.losses.update(loss.item())


def test(model, loader, opt, cur_rank):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            gt = data.y

            out = model(inputs)
            pred = out.max(dim=1)[1]

            pred_np = pred.cpu().numpy()
            target_np = gt.cpu().numpy()

            for cl in range(opt.n_classes):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)
                I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                Is[i, cl] = I
                Us[i, cl] = U

    ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    ious[np.isnan(ious)] = 1
    iou = np.mean(ious)
    
    if opt.phase == 'test':
        for cl in range(opt.n_classes):
            logging.info("===> mIOU for class {}: {}".format(cl, ious[cl]))

    opt.test_value = iou
    logging.info('TEST Epoch: [{}]\t mIoU: {:.4f}\t'.format(opt.epoch, opt.test_value))

def epochs(opt):
    logging.info('===> Creating dataloader ...')
    train_dataset = GeoData.S3DIS(opt.data_dir, opt.area, True, pre_transform=T.NormalizeScale())
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=opt.seed)
    train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler = train_sampler, num_workers=opt.n_gpus)
    test_dataset = GeoData.S3DIS(opt.data_dir, opt.area, train=False, pre_transform=T.NormalizeScale())
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=opt.seed)
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, sampler = test_sampler, num_workers=opt.n_gpus)
    opt.n_classes = train_loader.dataset.num_classes

    cur_rank = comm.get_local_rank()

    logging.info('===> Loading the network ...')
    model = DistributedDataParallel(CustomDenseDeepGCN(opt).to(cur_rank),device_ids=[cur_rank], output_device=cur_rank,broadcast_buffers=False).to(cur_rank)

    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    logging.info(model)

    logging.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(cur_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    logging.info('===> Init Metric ...')
    opt.losses = AverageMeter()
    opt.test_value = 0.

    logging.info('===> start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        train_sampler.set_epoch(opt.epoch)
        test_sampler.set_epoch(opt.epoch)
        logging.info('Epoch:{}'.format(opt.epoch))
        train(model, train_loader, optimizer, criterion, opt, cur_rank)
        if opt.epoch % opt.eval_freq == 0 and opt.eval_freq != -1:
            test(model, test_loader, opt, cur_rank)
        scheduler.step()
        if comm.is_main_process():
            # ------------------ save checkpoints
            # min or max. based on the metrics
            is_best = (opt.test_value < opt.best_value)
            opt.best_value = max(opt.test_value, opt.best_value)
            model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            save_checkpoint({
                'epoch': opt.epoch,
                'state_dict': model_cpu,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_value': opt.best_value,
            }, is_best, opt.ckpt_dir, opt.exp_name)
            # ------------------ tensorboard log
            info = {
                'loss': opt.losses.avg,
                'test_value': opt.test_value,
                'lr': scheduler.get_lr()[0]
            }
            writer.add_scalar('Train Loss', info['loss'], opt.epoch)
            writer.add_scalar('Test IOU', info['test_value'], opt.epoch)
            writer.add_scalar('lr', info['lr'], opt.epoch)

        logging.info('Saving the final model.Finish!')

def hola():
    print('Hola')

def main():
    opt = OptInit().get_args()
    '''
    This wrapper taken from detectron2 (https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py),
    creates n_gpus processes and launches epochs function on each of them.
    '''
    launch(
        epochs,
        num_gpus_per_machine=opt.n_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url='auto',
        args=(opt,)
    )
    #epochs(opt)
    
if __name__ == '__main__':
    main()