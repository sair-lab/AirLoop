from .lifelong import MASLoss, RKDLoss, EWCLoss


def get_ll_loss(args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None):
    if args.ll_method is None:
        return None
    elif args.ll_method.lower() == 'mas':
        loss = MASLoss(args=args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter)
    elif args.ll_method.lower() == 'ewc':
        loss = EWCLoss(args=args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter)
    elif args.ll_method.lower() == 'rkd':
        loss = RKDLoss(args=args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter)

    return loss
