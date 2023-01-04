from . import (dualcoop)

def build_model(cfg, args, classnames):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    model = dualcoop(cfg, classnames)
    network_name = model.network_name if hasattr(model, 'network_name') else cfg.MODEL.BACKBONE.NAME
    arch_name = "{dataset}-{arch_name}".format(
        dataset=cfg.DATASET.NAME, arch_name=network_name)
    # add setting info only in training

    arch_name += "{}".format('-' + args.prefix if args.prefix else "")
    if not args.evaluate:
        arch_name += "-{}-bs{}-e{}".format(cfg.OPTIM.LR_SCHEDULER, cfg.DATALOADER.TRAIN_X.BATCH_SIZE,  cfg.OPTIM.MAX_EPOCH)
    return model, arch_name