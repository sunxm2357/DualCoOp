import os
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models import build_model
from utils.validations import validate_zsl
from opts import arg_parser
from dataloaders import build_dataset
from utils.build_cfg import setup_cfg

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cfg = setup_cfg(args)

    test_split = cfg.DATASET.TEST_SPLIT
    test_gzsi_dataset = build_dataset(cfg, test_split, cfg.DATASET.ZS_TEST)
    test_gzsi_cls_id = test_gzsi_dataset.cls_id
    test_gzsl_split =  cfg.DATASET.TEST_GZSL_SPLIT
    test_unseen_dataset = build_dataset(cfg, test_gzsl_split, cfg.DATASET.ZS_TEST_UNSEEN)
    test_unseen_cls_id = test_unseen_dataset.cls_id
    test_gzsi_loader = torch.utils.data.DataLoader(test_gzsi_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                                                   shuffle=cfg.DATALOADER.TEST.SHUFFLE,
                                                   num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    test_unseen_loader = torch.utils.data.DataLoader(test_unseen_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                                                     shuffle=cfg.DATALOADER.TEST.SHUFFLE,
                                                     num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)


    classnames = test_gzsi_dataset.classnames

    model, arch_name = build_model(cfg, args, classnames)

    model.eval()

    if args.pretrained is not None and os.path.exists(args.pretrained):
        print('... loading pretrained weights from %s' % args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        epoch = checkpoint['epoch']
        model.load_state_dict(state_dict)
        print('Epoch: %d' % epoch)
    else:
        raise ValueError('args.pretrained is missing or its path does not exist')


    print('Evaluate with threshold %.2f' % args.thre)
    p_unseen, r_unseen, f1_unseen, mAP_unseen = validate_zsl(test_unseen_loader, model, args, test_unseen_cls_id)
    p_gzsl, r_gzsl, f1_gzsl, mAP_gzsl = validate_zsl(test_gzsi_loader, model, args, test_gzsi_cls_id)
    print('Test: [{}/{}]\t '
          ' P_unseen {:.2f} \t R_unseen {:.2f} \t F1_unseen {:.2f} \t mAP_unseen {:.2f}\t'
          ' P_gzsl {:.2f} \t R_gzsl {:.2f} \t F1_gzsl {:.2f} \t mAP_gzsl {:.2f}\t'
          .format(epoch, cfg.OPTIM.MAX_EPOCH, p_unseen, r_unseen, f1_unseen, mAP_unseen, p_gzsl, r_gzsl, f1_gzsl,
                  mAP_gzsl), flush=True)


if __name__ == '__main__':
    main()