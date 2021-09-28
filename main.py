#!/usr/bin/env python3

from pathlib import Path
import configargparse

from main_single import run


def main(args, extra_args):
    n_env = len(args.envs)
    if len(args.epochs) == 1:
        args.epochs = args.epochs * n_env
    else:
        assert args.method != 'joint' and len(args.epochs) == n_env

    out_dir = Path(args.out_dir)
    all_env_regex = '(' + '|'.join(args.envs) + ')'

    save_path = create_dir(out_dir / 'train') / 'airloop.pth'
    for i, (epoch, env) in enumerate(zip(args.epochs, args.envs)):

        if not args.skip_train:
            train_args = ['--task', 'train-joint' if args.method == 'joint' else 'train-seq']
            train_args += ['--include', all_env_regex if args.method == 'joint' else env]
            train_args += ['--epoch', str(epoch)]

            # load model saved from previous runx
            if i > 0:
                train_args += ['--load', str(save_path)]
            train_args += ['--save', str(save_path)]

            # weights loading for lifelong methods
            if args.method not in ['finetune', 'joint']:
                train_args += ['--ll-method', args.method]
                train_args += ['--ll-weight-dir', str(create_dir(out_dir / 'll-weights'))]
                if i > 0:
                    train_args += ['--ll-weight-load'] + args.envs[:i]

            run(train_args + extra_args)

        if args.method == 'joint':
            save_path = save_path.parent / (save_path.name + f'.epoch{epoch - 1}')
        else:
            save_path = save_path.parent / (save_path.name + (f".{env}.{epoch - 1}" if epoch > 1 else f".{env}"))

        if not args.skip_eval:
            eval_args = ['--task', 'eval', '--include', all_env_regex, '--load', str(save_path)]
            run(eval_args + extra_args)


def create_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    return directory


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    # meta
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    # launch
    parser.add_argument('--envs', type=str, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, nargs='+', default=[1])
    parser.add_argument('--method', type=str, required=False, default='finetune',
                        choices=['finetune', 'si', 'ewc', 'kd', 'rkd', 'mas', 'rmas', 'airloop', 'joint'])

    parserd_args, unknown_args = parser.parse_known_args()

    main(parserd_args, unknown_args)
