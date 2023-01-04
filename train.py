import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models import build_model
from utils.validations import validate
from opts import arg_parser
from dataloaders import build_dataset
from utils.build_cfg import setup_cfg
from dassl.optim import build_lr_scheduler
from utils.trainers import train_coop
from utils.helper import save_checkpoint


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cfg = setup_cfg(args)

    # building the train and val dataloaders
    train_split = cfg.DATASET.TRAIN_SPLIT
    val_split = cfg.DATASET.VAL_SPLIT
    test_split = cfg.DATASET.TEST_SPLIT
    train_dataset = build_dataset(cfg, train_split)
    val_dataset = build_dataset(cfg, val_split)
    test_dataset = build_dataset(cfg, test_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                                             shuffle=cfg.DATALOADER.TRAIN_X.SHUFFLE,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.DATALOADER.VAL.BATCH_SIZE,
                                             shuffle=cfg.DATALOADER.VAL.SHUFFLE,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                                             shuffle=cfg.DATALOADER.TEST.SHUFFLE,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    classnames = val_dataset.classnames

    # build the model
    model, arch_name = build_model(cfg, args, classnames)
    # build the optimizer and lr_scheduler
    # optim = build_optimizer(model.prompt_learner, cfg.OPTIM)
    try:
        prompt_params = model.prompt_params()
    except:
        prompt_params = model.module.prompt_params()
    prompt_group = {'params': prompt_params}
    print('num of params in prompt learner: ', len(prompt_params))
    sgd_polices = [prompt_group]
    if cfg.TRAINER.FINETUNE_BACKBONE:
        try:
            backbone_params = model.backbone_params()
        except:
            backbone_params = model.module.backbone_params()
        print('num of params in backbone: ', len(backbone_params))
        base_group = {'params': backbone_params, 'lr': cfg.OPTIM.LR * cfg.OPTIM.BACKBONE_LR_MULT}
        sgd_polices.append(base_group)

    if cfg.TRAINER.FINETUNE_ATTN:
        try:
            attn_params = model.attn_params()
        except:
            attn_params = model.module.attn_params()
        print('num of params in attn layer: ', len(attn_params))
        attn_group = {'params': attn_params, 'lr': cfg.OPTIM.LR * cfg.OPTIM.ATTN_LR_MULT}
        sgd_polices.append(attn_group)


    optim = torch.optim.SGD(sgd_polices, lr=cfg.OPTIM.LR,
                                momentum=cfg.OPTIM.MOMENTUM,
                                weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                                dampening=cfg.OPTIM.SGD_DAMPNING,
                                nesterov=cfg.OPTIM.SGD_NESTEROV)

    sched = build_lr_scheduler(optim, cfg.OPTIM)
    log_folder = os.path.join(cfg.OUTPUT_DIR, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logfile_path = os.path.join(log_folder, 'log.log')
    if os.path.exists(logfile_path):
        logfile = open(logfile_path, 'a')
    else:
        logfile = open(logfile_path, 'w')

    # logging out some useful information on screen and into log file
    command = " ".join(sys.argv)
    print(command, flush=True)
    print(args, flush=True)
    print(model, flush=True)
    print(cfg, flush=True)
    print(command, file=logfile, flush=True)
    print(args, file=logfile, flush=True)
    print(cfg, file=logfile, flush=True)


    if not args.auto_resume:
        print(model, file=logfile, flush=True)

    if args.auto_resume:
        # checkpoint_path = os.path.join(log_folder, 'checkpoint.pth.tar')
        # if os.path.exists(checkpoint_path):
        args.resume = os.path.join(log_folder, 'checkpoint.pth.tar')

    best_mAP = 0
    args.start_epoch = 0
    if args.resume is not None:
        if os.path.exists(args.resume):
            print('... loading pretrained weights from %s' % args.resume)
            print('... loading pretrained weights from %s' % args.resume, file=logfile, flush=True)
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            # TODO: handle distributed version
            best_mAP = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            sched.load_state_dict(checkpoint['scheduler'])

    for epoch in range(args.start_epoch, cfg.OPTIM.MAX_EPOCH):
        batch_time, losses, mAP_batches = train_coop(train_loader, [val_loader],  model, optim, sched, args, cfg, epoch)
        print('Train: [{0}/{1}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.2f} \t'
              'mAP {mAP_batches.avg:.2f}'.format(
            epoch + 1, cfg.OPTIM.MAX_EPOCH, batch_time=batch_time,
            losses=losses, mAP_batches=mAP_batches), flush=True)

        print('Train: [{0}/{1}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.2f} \t'
              'mAP {mAP_batches.avg:.2f}'.format(
            epoch + 1, cfg.OPTIM.MAX_EPOCH, batch_time=batch_time,
            losses=losses, mAP_batches=mAP_batches), file=logfile, flush=True)

        if (epoch + 1) % args.val_every_n_epochs == 0 or epoch == args.stop_epochs - 1:
            p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = validate(val_loader, model, args)
            print('Test: [{}/{}]\t '
                  ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
                  .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score), flush=True)
            print('Test: [{}/{}]\t '
                  ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
                  .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score),
                  file=logfile, flush=True)

            is_best = mAP_score > best_mAP
            if is_best:
                best_mAP = mAP_score
            save_dict = {'epoch': epoch + 1,
                         'arch': arch_name,
                         'state_dict': model.state_dict(),
                         'best_mAP': best_mAP,
                         'optimizer': optim.state_dict(),
                         'scheduler': sched.state_dict()
                         }
            save_checkpoint(save_dict, is_best, log_folder)

    print('Evaluating the best model', flush=True)
    print('Evaluating the best model', file=logfile, flush=True)

    print('Evaluate with threshold %.2f' % args.thre, flush=True)
    print('Evaluate with threshold %.2f' % args.thre, file=logfile, flush=True)

    best_checkpoints = os.path.join(log_folder, 'model_best.pth.tar')
    print('... loading pretrained weights from %s' % best_checkpoints, flush=True)
    print('... loading pretrained weights from %s' % best_checkpoints, file=logfile, flush=True)
    checkpoint = torch.load(best_checkpoints, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch']
    p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = validate(test_loader, model, args)
    print('Test: [{}/{}]\t '
          ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
          .format(best_epoch, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score))
    print('Test: [{}/{}]\t '
          ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
          .format(best_epoch, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score),
          file=logfile, flush=True)


if __name__ == '__main__':
    main()