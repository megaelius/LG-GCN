
import os

#os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda-11.1/bin64:/usr/local/cuda-11.2/bin64'

import numpy as np
import torch
import torch.multiprocessing as mp
import torch_geometric.datasets as GeoData
from modelnet40 import ModelNet40
from torch.utils.data import DataLoader, Sampler
import torch_geometric.transforms as T
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from config import OptInit
from architecture import ClassificationGraphNN, ClassificationGraphNN2
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
import logging
from tqdm import tqdm
from parallel_wrapper import launch
import comm
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix
from gcn_lib.dense import pairwise_distance
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dist2distloss(feats1,feats2):
    f1 = feats1.transpose(2, 1).squeeze(-1)
    f2 = feats2.transpose(2, 1).squeeze(-1)
    d1 = pairwise_distance(f1)/f1.shape[2]
    d2 = pairwise_distance(f2)/f2.shape[2]
    crit = torch.nn.MSELoss(reduction='sum')
    return crit(d1,d2)

def stressloss(feats1,feats2):
    f1 = feats1.transpose(2, 1).squeeze(-1)
    f2 = feats2.transpose(2, 1).squeeze(-1)
    d1 = torch.sqrt(pairwise_distance(f1)/f1.shape[2] + 1e-6)
    d2 = torch.sqrt(pairwise_distance(f2)/f2.shape[2] + 1e-6)
    crit = torch.nn.MSELoss(reduction='sum')
    return crit(d1,d2)

def scaledstressloss(feats1,feats2):
    f1 = feats1.transpose(2, 1).squeeze(-1)
    f2 = feats2.transpose(2, 1).squeeze(-1)
    d1squared = pairwise_distance(f1)
    d2squared = pairwise_distance(f2)
    #We add 1 to avoid problems with squared root gradients, this doesn't affect as we compute the difference
    #d1 = torch.sqrt(d1squared/f1.shape[2] + 1)
    #d2 = torch.sqrt(d2squared/f2.shape[2] + 1)
    d1 = torch.sqrt(d1squared + 1)
    d2 = torch.sqrt(d2squared + 1)
    crit = torch.nn.MSELoss(reduction='none')
    #print(d1.shape, ((d1squared.view(f1.shape[0],-1)).sum(1, keepdim=True).unsqueeze(-1)).shape)
    #Shapes are B,N,N and B,1,1
    scaled_se = crit(d1,d2)/((d1squared.view(f1.shape[0],-1)).sum(1, keepdim=True).unsqueeze(-1))
    #print(scaled_se.shape)
    flat_scaled_se = scaled_se.view(f1.shape[0],-1)
    #print(flat_scaled_se.shape)
    return flat_scaled_se.sum(dim=1).sum(dim=0)


def train(model, train_loader, optimizer, criterion, opt, cur_rank):
    model.train()
    total_loss = 0
    total_d2d_loss = 0
    total_ce_loss = 0

    targets = []
    preds = []
    with tqdm(train_loader) as tqdm_loader:
        for i, (data,label) in enumerate(tqdm_loader):
            opt.iter += 1
            desc = 'Epoch:{}  Iter:{}  [{}/{}]'\
                .format(opt.epoch, opt.iter, i + 1, len(train_loader))
            tqdm_loader.set_description(desc)

            inputs = data.transpose(2, 1).unsqueeze(3).to(opt.device)
            gt = label.to(opt.device)
            # ------------------ zero, output, loss
            optimizer.zero_grad()
            out, graph_feats = model(inputs)
            #Cros Entropy
            loss = criterion(out, gt)
            total_ce_loss += loss.item()

            #L2 regularization of Graph features
            l2_crit = torch.nn.L1Loss(reduction='sum')
            reg_loss = 0
            for name, param in model.named_parameters():
                if name[:16] == 'module.graph_mlp':
                    zero = torch.zeros_like(param)
                    reg_loss += l2_crit(param,zero)
            factor_l2 = opt.graph_l2reg
            #Divergence of distance matrices
            factor_dist2dist = opt.d2d_weight
            d2d_loss = scaledstressloss(inputs.to(opt.device), graph_feats)
            total_d2d_loss += d2d_loss.item()

            #Update loss
            loss += factor_l2 * reg_loss + factor_dist2dist*d2d_loss
            total_loss+=loss.item()
            # ------------------ optimization
            loss.backward()
            optimizer.step()

            target_np = gt.cpu().numpy()
            pred = out.max(dim=1)[1]
            pred_np = pred.cpu().numpy()

            targets += list(target_np.ravel())
            preds += list(pred_np.ravel())

    acc = accuracy_score(targets, preds)

    return total_loss, total_d2d_loss, total_ce_loss, acc


def test(model, loader, criterion, opt, cur_rank):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))
    total_loss = 0
    total_d2d_loss = 0
    total_ce_loss = 0

    targets = []
    preds = []

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(loader)):
            inputs = data.transpose(2, 1).unsqueeze(3).to(opt.device)
            gt = label.to(opt.device)

            out, graph_feats = model(inputs)
            loss = criterion(out, gt.to(opt.device))
            total_ce_loss += loss.item()

            #Divergence of distance matrices
            factor_dist2dist = opt.d2d_weight
            d2d_loss = scaledstressloss(inputs.to(opt.device), graph_feats)
            total_d2d_loss += d2d_loss.item()

            #Update loss
            loss += factor_dist2dist*d2d_loss
            total_loss+=loss.item()
            pred = out.max(dim=1)[1]

            target_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()

            targets += list(target_np.ravel())
            preds += list(pred_np.ravel())

    acc = accuracy_score(targets, preds)

    logging.info('TEST Epoch: [{}]\t Acc: {:.4f}\t'.format(opt.epoch, acc))
    return targets, preds, total_loss, total_d2d_loss, total_ce_loss, acc

def epochs(opt):
    logging.info('===> Creating dataloader ...')
    #train_dataset = GeoData.ModelNet(opt.data_dir, name = '40', train = True, pre_transform=T.NormalizeScale())
    #train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=opt.seed)
    #train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler = train_sampler, num_workers=opt.n_gpus)
    #test_dataset = GeoData.ModelNet(opt.data_dir, name = '40', train=False, pre_transform=T.NormalizeScale())
    #test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=opt.seed)
    #test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, sampler = test_sampler, num_workers=opt.n_gpus)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_dataset = ModelNet40(1024, 'train')
    train_generator = torch.Generator()
    train_generator.manual_seed(opt.seed)
    if opt.n_gpus > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=opt.seed)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, generator= train_generator, sampler = train_sampler, num_workers=opt.n_gpus, drop_last=False,worker_init_fn=seed_worker)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, generator=train_generator, num_workers=4, drop_last=False,worker_init_fn=seed_worker)

    test_dataset = ModelNet40(1024, 'test')
    test_generator = torch.Generator()
    test_generator.manual_seed(opt.seed)
    if opt.n_gpus > 1:
        test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=opt.seed)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, generator = test_generator, sampler = test_sampler, num_workers=opt.n_gpus, drop_last=False,worker_init_fn=seed_worker)
    else:
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, generator=test_generator, num_workers=4, drop_last=False,worker_init_fn=seed_worker)

    LABELS = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox']

    opt.n_classes = 40

    cur_rank = comm.get_local_rank()

    #torch.use_deterministic_algorithms(True)

    logging.info('===> Loading the network ...')
    if opt.n_gpus > 1:
        model = DistributedDataParallel(ClassificationGraphNN2(opt).to(cur_rank),device_ids=[cur_rank], output_device=cur_rank,broadcast_buffers=False).to(cur_rank)
    else:
        model = ClassificationGraphNN2(opt).to(cur_rank)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    logging.info(model)
    if comm.is_main_process():
        wandb.init(project="MODELNET40")
        wandb.run.name = opt.exp_name
        wandb.watch(model,log_freq=100,log="all")

    logging.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(cur_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay = opt.decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    logging.info('===> start training ...')
    epochs_frozen = 0
    freeze_graph = True
    for _ in range(opt.epoch, opt.total_epochs):
        '''
        filtered_parameters = []
        if epochs_frozen == 5:
            freeze_graph = not freeze_graph
            epochs_frozen = 0
        for name, param in model.named_parameters():
            #filter(lambda t: ( t[0][:16] == 'module.graph_mlp') or (not bool(opt.epoch%2) and t[0][:16] != 'module.graph_mlp'), model.parameters())
            if (freeze_graph and name[:16] == 'module.graph_mlp') or (not freeze_graph and name[:16] != 'module.graph_mlp'):
                print(name)
                filtered_parameters.append(param)
        optimizer = torch.optim.Adam(filtered_parameters)
        '''

        opt.epoch += 1
        if opt.n_gpus > 1:
            train_sampler.set_epoch(opt.epoch)
            test_sampler.set_epoch(opt.epoch)
        logging.info('Epoch:{}'.format(opt.epoch))
        train_loss, train_d2d_loss, train_ce_loss, train_acc = train(model, train_loader, optimizer, criterion, opt, cur_rank)
        if opt.epoch % opt.eval_freq == 0 and opt.eval_freq != -1:
            test_targets, test_preds, test_loss, test_d2d_loss, test_ce_loss, test_acc = test(model, test_loader, criterion, opt, cur_rank)
        scheduler.step()
        if comm.is_main_process():
            # ------------------ save checkpoints
            # min or max. based on the metrics
            is_best = (test_acc < opt.best_value)
            opt.best_value = max(test_acc, opt.best_value)
            model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            save_checkpoint({
                'epoch': opt.epoch,
                'state_dict': model_cpu,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_value': opt.best_value,
            }, is_best, opt.ckpt_dir, opt.exp_name)
            matrix = confusion_matrix(test_targets, test_preds)
            per_class_acc = matrix.diagonal()/matrix.sum(axis=1)
            wandb_dict = {'Train/loss': train_loss,
                       'Val/loss': test_loss,
                       'Train/Accuracy':train_acc,
                       'Val/Accuracy':test_acc,
                       'Train/d2d_loss':train_d2d_loss,
                       'Val/d2d_loss':test_d2d_loss,
                       'Train/ce_loss':train_ce_loss,
                       'Val/ce_loss':test_ce_loss,
                       'lr':scheduler.get_last_lr()[0]}
            for i, c in enumerate(LABELS):
                wandb_dict[f'Test_acc/{c}'] = per_class_acc[i]
            wandb_dict[f'Test_acc/MEAN'] = sum(per_class_acc)/opt.n_classes
            # ------------------ tensorboard log
            wandb.log(wandb_dict, step=opt.epoch)

        logging.info('Saving the final model.Finish!')
        epochs_frozen += 1

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
