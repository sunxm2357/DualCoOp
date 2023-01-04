import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import time
from utils.helper import AverageMeter, mAP
from utils.validations import validate, validate_zsl
from utils.asymmetric_loss import AsymmetricLoss, AsymmetricLoss2, AsymmetricLoss3
from torch.cuda.amp import autocast


def train_classic_fc(data_loader, val_loader, model, optim, sched, scaler, args, cfg, epoch):
    batch_time = AverageMeter()
    mAP_batches = AverageMeter()
    losses = AverageMeter()
    Softmax = torch.nn.Softmax(dim=1)

    # switch to evaluate mode
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model.fc.train()
        if cfg.TRAINER.FINETUNE:
            model.train()
    else:
        model.module.fc.train()
        if cfg.TRAINER.FINETUNE:
            model.train()

    criterion = AsymmetricLoss(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)

    end = time.time()
    for i,  (images, target) in enumerate(data_loader):
        target = target.max(dim=1)[0]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        images = images.to(device)
        target = target.to(device)

        # compute output
        with autocast():
            output = model(images)
        loss = args.loss_w * criterion(output, target)

        model.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optim)
        scaler.update()

        sched.step()

        losses.update(loss.item(), images.size(0))
        pred = Softmax(output.detach())[:, 1, :]
        mAP_value = mAP(target.cpu().numpy(), pred.cpu().numpy())
        mAP_batches.update(mAP_value, images.size(0))
        batch_time.update(time.time()-end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
                  'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f})'.format(
                i, len(data_loader), batch_time=batch_time,
                losses=losses, mAP_batches=mAP_batches), flush=True)

        if args.val_freq_in_epoch != -1 and (i + 1) % args.val_freq_in_epoch == 0:
            p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = validate(val_loader, model, args)
            print('Test: [{}/{}]\t '
                  ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
                  .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score), flush=True)

    return batch_time, losses, mAP_batches


def train_coop(data_loader, val_loaders, model, optim, sched, args, cfg, epoch, cls_id=None):
    batch_time = AverageMeter()
    mAP_batches = AverageMeter()
    losses = AverageMeter()
    Softmax = torch.nn.Softmax(dim=1)
    Sig = torch.nn.Sigmoid()

    if cls_id is not None:
        num_train_cls = len(cls_id['train'])
    # # switch to evaluate mode
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.image_encoder.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.image_encoder.train()
    else:
        model.module.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.module.image_encoder.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.module.image_encoder.train()

    criterion = AsymmetricLoss(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)
    criterion2 = AsymmetricLoss2(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)
    criterion3 = AsymmetricLoss3(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)

    end = time.time()
    for i,   (images, target) in enumerate(data_loader):
        target = target.max(dim=1)[0]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        images = images.to(device)
        target = target.to(device)
        if cls_id is not None:
            if num_train_cls > args.num_train_cls:
                batch_cls_id = torch.randperm(num_train_cls).cpu().tolist()[:args.num_train_cls]
                batch_cls_id_input = [cls_id['train'][a] for a in batch_cls_id]
            else:
                batch_cls_id_input = cls_id['train']
        else:
            batch_cls_id_input = None

        # compute output
        with autocast():
            output = model(images, batch_cls_id_input)
        # loss = args.loss_w * criterion(output, target)
        if cls_id is not None:
            # output = output[:, :, cls_id['train']]
            # target = target[:, cls_id['train']]
            target = target[:, batch_cls_id_input]
        if output.dim() == 3:
            loss = args.loss_w * criterion(output, target)
        elif args.single_prompt == 'pos':
            loss = args.loss_w * criterion2(output, target)
        elif args.single_prompt == 'neg':
            loss = args.loss_w * criterion3(output, target)
        else:
            raise ValueError

        # update the network
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss.item(), images.size(0))
        if output.dim() == 3:
            pred = Softmax(output.detach())[:, 1, :]
        else:
            pred = Sig(output.detach())
        mAP_value = mAP(target.cpu().numpy(), pred.cpu().numpy())
        mAP_batches.update(mAP_value, images.size(0))
        batch_time.update(time.time()-end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
                  'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f})'.format(
                i, len(data_loader), batch_time=batch_time,
                losses=losses, mAP_batches=mAP_batches), flush=True)

        if args.val_freq_in_epoch != -1 and (i + 1) % args.val_freq_in_epoch == 0:
            if len(val_loaders) == 1:
                p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = validate(val_loaders[0], model, args)
                print('Test: [{}/{}]\t '
                      ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
                      .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score), flush=True)
            elif len(val_loaders) == 2:
                p_unseen, r_unseen, f1_unseen, mAP_unseen = validate_zsl(val_loaders[0], model, args,
                                                                         cls_id['val_unseen'])
                p_gzsl, r_gzsl, f1_gzsl, mAP_gzsl = validate_zsl(val_loaders[1], model, args, cls_id['val_gzsi'])
                print('Test: [{}/{}]\t '
                      ' P_unseen {:.2f} \t R_unseen {:.2f} \t F1_unseen {:.2f} \t mAP_unseen {:.2f}\t'
                      ' P_gzsl {:.2f} \t R_gzsl {:.2f} \t F1_gzsl {:.2f} \t mAP_gzsl {:.2f}\t'
                      .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_unseen, r_unseen, f1_unseen, mAP_unseen, p_gzsl, r_gzsl,
                              f1_gzsl, mAP_gzsl), flush=True)
            else:
                raise ValueError

    sched.step()

    return batch_time, losses, mAP_batches