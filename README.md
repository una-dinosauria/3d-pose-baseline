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

* [h5py](http://www.h5py.org/)
* [tensorflow](https://www.tensorflow.org/) 1.0 or later

### First of all
1. Watch our video: https://youtu.be/Hmi3Pd9x1BE
2. Clone this repository and get the data. We provide the [Human3.6M](http://vision.imar.ro/human3.6m/description.php) dataset in 3d points, camera parameters to produce ground truth 2d detections, and [Stacked Hourglass](https://github.com/anewell/pose-hg-demo) detections.

```bash
git clone https://github.com/una-dinosauria/3d-pose-baseline.git
cd 3d-pose-baseline
mkdir data
cd data
wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
unzip h36m.zip
rm h36m.zip
cd ..
```

### Quick demo

For a quick demo, you can train for one epoch and visualize the results. To train, run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 1`

This should take about <5 minutes to complete on a GTX 1080, and give you around 75 mm of error on the test set.

Now, to visualize the results, simply run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 1 --sample --load 24371`

This will produce a visualization similar to this:

![Visualization example](/imgs/viz_example.png?raw=1)

### Training

To train a model with clean 2d detections, run:

<!-- `python src/predict_3dpose.py --camera_frame --residual` -->
`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise`

This corresponds to Table 2, bottom row. `Ours (GT detections) (MA)`

To train on Stacked Hourglass detections, run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh`

This corresponds to Table 2, next-to-last row. `Ours (SH detections) (MA)`

On a GTX 1080 GPU, this takes <8 ms for forward+backward computation, and
<6 ms for forward-only computation per batch of 64.

### Pre-trained model

We also provide a model pre-trained on Stacked-Hourglass detections, available through [google drive](https://drive.google.com/file/d/0BxWzojlLp259MF9qSFpiVjl0cU0/view?usp=sharing)

To test the model, decompress the file at the top level of this project, and call

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --sample --load 4874200`

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

### License
MIT
