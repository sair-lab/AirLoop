from .lifelong import MASLoss, KDLoss, EWCLoss, SILoss, CompoundLifelongLoss


def get_ll_loss(args, writer=None, viz=None, viz_start=float('inf'), viz_freq=200, counter=None):
    if args.ll_method is None:
        return None
    assert len(args.ll_method) == len(args.ll_strength)

    # create each lifelong loss
    losses = []
    extra_kwargs = {}
    for method, strength in zip(args.ll_method, args.ll_strength):
        method = method.lower()
        if method == 'mas':
            extra_kwargs['relational'] = False
            loss_class = MASLoss
        elif method == 'rmas':
            loss_class = MASLoss
        elif method == 'ewc':
            loss_class = EWCLoss
        elif method == 'si':
            loss_class = SILoss
        elif method == 'kd':
            extra_kwargs['relational'] = False
            loss_class = KDLoss
        elif method == 'rkd':
            loss_class = KDLoss
        elif method == 'crkd':
            loss_class = KDLoss
            extra_kwargs['last_only'] = False
        else:
            raise ValueError(f'Unrecognized lifelong method: {method}')

        losses.append(loss_class(args=args, writer=writer, viz=viz, viz_start=viz_start, viz_freq=viz_freq, counter=counter, lamb=strength, **extra_kwargs))

    return CompoundLifelongLoss(*losses)
