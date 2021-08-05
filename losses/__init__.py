from .lifelong import MASLoss, RKDLoss, EWCLoss, SILoss


def get_ll_loss(args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None, lamb=1):
    if args.ll_method is None:
        return None
    elif args.ll_method.lower() == 'mas':
        loss_class = MASLoss
    elif args.ll_method.lower() == 'ewc':
        loss_class = EWCLoss
    elif args.ll_method.lower() == 'si':
        loss_class = SILoss
    elif args.ll_method.lower() == 'rkd':
        loss_class = RKDLoss
    else:
        raise ValueError(f'Unrecognized lifelong method: {args.ll_method}')

    return loss_class(args=args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=lamb)
