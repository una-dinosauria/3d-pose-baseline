## 3d-pose-baseline

This is the code for the paper

Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little.
_A simple yet effective baseline for 3d human pose estimation._
In ICCV, 2017. https://arxiv.org/pdf/1705.03098.pdf.

The code in this repository was mostly written by
[Julieta Martinez](https://github.com/una-dinosauria),
[Rayat Hossain](https://github.com/rayat137) and
[Javier Romero](https://github.com/libicocco).

We provide a strong baseline for 3d human pose estimation that also sheds light
on the challenges of current approaches. Our model is lightweight and we strive
to make our code transparent, compact, and easy-to-understand.

### Dependencies

* Python ≥ 3.5
* [cdflib](https://github.com/MAVENSDC/cdflib)
* [tensorflow](https://www.tensorflow.org/) 1.0 or later

### First of all
1. Watch our video: https://youtu.be/Hmi3Pd9x1BE

2. Clone this repository

```bash
git clone https://github.com/una-dinosauria/3d-pose-baseline.git
cd 3d-pose-baseline
mkdir -p data/h36m/
```

3. Get the data

Go to http://vision.imar.ro/human3.6m/, log in, and download the `D3 Positions` files for subjects `[1, 5, 6, 7, 8, 9, 11]`,
and put them under the folder `data/h36m`. Your directory structure should look like this
```bash
src/
README.md
LICENCE
...
data/
  └── h36m/
    ├── Poses_D3_Positions_S1.tgz
    ├── Poses_D3_Positions_S11.tgz
    ├── Poses_D3_Positions_S5.tgz
    ├── Poses_D3_Positions_S6.tgz
    ├── Poses_D3_Positions_S7.tgz
    ├── Poses_D3_Positions_S8.tgz
    └── Poses_D3_Positions_S9.tgz
```

Now, move to the data folder, and uncompress all the data

```bash
cd data/h36m/
for file in *.tgz; do tar -xvzf $file; done
```

Finally, download the `code-v1.2.zip` file, unzip it, and copy the `metadata.xml` file under `data/h36m/`

Now, your data directory should look like this:

```bash
data/
  └── h36m/
    ├── metadata.xml
    ├── S1/
    ├── S11/
    ├── S5/
    ├── S6/
    ├── S7/
    ├── S8/
    └── S9/

```

There is one little fix we need to run for the data to have consistent names:

```bash
mv h36m/S1/MyPoseFeatures/D3_Positions/TakingPhoto.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/Photo.cdf

mv h36m/S1/MyPoseFeatures/D3_Positions/TakingPhoto\ 1.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/Photo\ 1.cdf

mv h36m/S1/MyPoseFeatures/D3_Positions/WalkingDog.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/WalkDog.cdf

mv h36m/S1/MyPoseFeatures/D3_Positions/WalkingDog\ 1.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/WalkDog\ 1.cdf
```

And you are done!

Please note that we are currently not supporting SH detections anymore, only training from GT 2d detections is possible now.

### Quick demo

For a quick demo, you can train for one epoch and visualize the results. To train, run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 1`

This should take about <5 minutes to complete on a GTX 1080, and give you around 56 mm of error on the test set.

Now, to visualize the results, simply run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 1 --sample --load 24371`

This will produce a visualization similar to this:

![Visualization example](/imgs/viz_example.png?raw=1)

### Training

To train a model with clean 2d detections, run:

<!-- `python src/predict_3dpose.py --camera_frame --residual` -->
`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise`

This corresponds to Table 2, bottom row. `Ours (GT detections) (MA)`

<!--
To train on Stacked Hourglass detections, run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh`

This corresponds to Table 2, next-to-last row. `Ours (SH detections) (MA)`

On a GTX 1080 GPU, this takes <8 ms for forward+backward computation, and
<6 ms for forward-only computation per batch of 64.
-->

<!--
### Pre-trained model

We also provide a model pre-trained on ground truth 2d detections, available through [google drive](https://drive.google.com/file/d/0BxWzojlLp259MF9qSFpiVjl0cU0/view?usp=sharing).

To test the model, decompress the file at the top level of this project, and call

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 200 --sample --load 4874200`
-->

<!--
### Fine-tuned stacked-hourglass detections

You can find the detections produced by Stacked Hourglass after fine-tuning on the H3.6M dataset on [google drive](https://drive.google.com/open?id=0BxWzojlLp259S2FuUXJ6aUNxZkE).
-->

### Citing

If you use our code, please cite our work

```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

### Other implementations

* [Pytorch](https://github.com/weigq/3d_pose_baseline_pytorch) by [@weigq](https://github.com/weigq)
* [MXNet/Gluon](https://github.com/lck1201/simple-effective-3Dpose-baseline) by [@lck1201](https://github.com/lck1201)

### Extensions

* [@ArashHosseini](https://github.com/ArashHosseini) maintains [a fork](https://github.com/ArashHosseini/3d-pose-baseline) for estimating 3d human poses using the 2d poses estimated by either [OpenPose](https://github.com/ArashHosseini/openpose) or [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) as input.

### License
MIT
