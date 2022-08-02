# AirLoop

This repo contains the source code for paper:

[Dasong Gao](http://theairlab.org/team/dasongg/), [Chen Wang](https://chenwang.site), [Sebastian Scherer](http://theairlab.org/team/sebastian/). "[AirLoop: Lifelong Loop Closure Detection](https://arxiv.org/pdf/2109.08975)." International Conference on Robotics and Automation (ICRA), 2022.

<p align="center">
  <a href="https://youtu.be/Gr9i5ONNmz0">
    <img src="assets/images/motivation.gif" alt="Watch on YouTube">
  </a>
</p>

## Demo

Examples of loop closure detection on each dataset. Note that our model is able to handle cross-environment loop closure detection despite only trained in individual environments sequentially:
<p align="center">
  <img src="assets/images/all-datasets.png">
</p>

Improved loop closure detection on TartanAir after extended training:
<p align="center">
  <img src="assets/images/tartanair-ll.gif">
</p>

## Usage
### Dependencies

 - Python >= 3.5
 - PyTorch < 1.8
 - OpenCV >= 3.4
 - NumPy >= 1.19
 - Matplotlib
 - ConfigArgParse
 - PyYAML
 - tqdm

### Data
We used the following subsets of datasets in our expriments:
 - [TartanAir](https://theairlab.org/tartanair-dataset/), download with [tartanair_tools](https://github.com/castacks/tartanair_tools)
   - Train/Test: `abandonedfactory_night`, `carwelding`, `neighborhood`, `office2`, `westerndesert`;
 - [RobotCar](https://robotcar-dataset.robots.ox.ac.uk/), download with [RobotCarDataset-Scraper](https://github.com/mttgdd/RobotCarDataset-Scraper)
   - Train: `2014-11-28-12-07-13`, `2014-12-10-18-10-50`, `2014-12-16-09-14-09`; 
   - Test: `2014-06-24-14-47-45`, `2014-12-05-15-42-07`, `2014-12-16-18-44-24`;
 - [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/), download with [gdown](https://github.com/wkentaro/gdown) from [Google Drive](https://drive.google.com/drive/folders/1SmrDOeUgBnJbpW187VFWxGjS7XdbZK5t)
   - Train/Test: All four seasons with recommended splits.

The datasets are aranged as follows:
```
$DATASET_ROOT/
├── tartanair/
│   ├── abandonedfactory_night/
|   |   ├── Easy/
|   |   |   └── ...
│   │   └── Hard/
│   │       └── ...
│   └── ...
├── robotcar/
│   ├── train/
│   │   ├── 2014-11-28-12-07-13/
│   │   └── ...
│   └── test/
│       ├── 2014-06-24-14-47-45/
│       └── ...
└── nordland/
    ├── train/
    │   ├── fall_images_train/
    │   └── ...
    └── test/
        ├── fall_images_test/
        └── ...
```

> **Note**: For TartanAir, only `<ENVIRONMENT>/<DIFFICULTY>/<image|depth>_left.zip` is required. After `unzip`ing downloaded zip files, make sure to remove the duplicate `<ENVIRONMENT>` directory level (`tartanair/abandonedfactory/abandonedfactory/Easy/...` -> `tartanair/abandonedfactory/Easy/...`).

### Configuration
The following values in [`config/config.yaml`](config/config.yaml) need to be set:
 - `dataset-root`: The parent directory to all datasets (`$DATASET_ROOT` above);
 - `catalog-dir`: An (initially empty) directory for caching processed dataset index;
 - `eval-gt-dir`: An (initially empty) directory for groundtruth produced during evaluation.

### Commandline
The following command trains the model with the specified method on TartanAir with default configuration and evaluate the performance:

```sh
$ python main.py --method <finetune/si/ewc/kd/rkd/mas/rmas/airloop/joint>
```
    
Extra options*:
 - `--dataset <tartanair/robotcar/nordland>`: dataset to use.
 - `--envs <LIST_OF_ENVIRONMENTS>`: order of environments.**
 - `--epochs <LIST_OF_EPOCHS>`: number of epochs to train in each environment.**
 - `--eval-save <PATH>`: save path for predicted pairwise similarities generated during evaluation.
 - `--out-dir <DIR>`: output directory for model checkpoints and importance weights.
 - `--log-dir <DIR>`: Tensorboard `logdir`.
 - `--skip-train`: perform evaluation only.
 - `--skip-eval`: perform training only.

\* See [`main_single.py`](main_single.py) for more settings.<br>
\** See [`main.py`](main.py) for defaults.

Evaluation results (R@100P in each environment) will be logged to console. `--eval-save` can be specified to save the predicted similarities in `.npz` format.
