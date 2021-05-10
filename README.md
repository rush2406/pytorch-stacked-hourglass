# Stacked Hourglass Network for Human pose estimation

Stacked hourglass networks have been very popular for human pose estimation. The proposed structure involves stacking 8 hourglasses. The hourglass has an encoder-decoder structure thus effectively combining local and global level information. Furthermore, intermediate supervision has been used between the hourglasses.

## Dataset

We have used the MPII dataset. It can be downloaded from [here](http://human-pose.mpi-inf.mpg.de/#download).

## Modifications

As part of our assignment, we tried to perform some modifications to reduce the training & inference time while maintaining comparable accuracy to the original 8-stacks hourglass model. The modifications and observed results can be summarized as follows (run for 50 epochs on the hardware mentioned below) - 

| Model         | Training Time (min/epoch) | Validation accuracy  | Inference time (min)
| ------------- |---------------------------| ---------------------| ---------------------|
| 1) original      | 18             | 83.18                |  1.2             |
| 2) 64 channels      | 14.3             | 80.48                |  1.03             |
| 3) 6 stacks      | 14.4             | 83.13                |  1.05             |
| 4) 6 stacks 64 channels 4 residual blocks      | 14.3             | 80.73                |  1.08             |
| 5) 8 stacks 64 channels, instance norm      | 14.7             | 78.83                |  1.1             |
| 6) 8 stacks, depth [2x3,3x3,4x2] 64 channels     | 14.3             | 80.68                |  1.1             |
| 7) 8 stacks, depth [4x2,3x4,2x2] 64 channels     | 14.8             | 81.47                |  1.15             |
| 8) 4 blocks {of Conv, ReLU, BatchNorm } in each bottleneck module      | 14.6             | 84.23                |  1.06   |

## Example scripts


### Evaluation on the MPII validation set

Architecture can be hg1, hg2, hg8.

```bash
$ python scripts/evaluate_mpii.py --arch=hg2 --image-path=/path/to/mpii/images --model-file=/path/to/saved/model_checkpoint
```

Output:

```
Final validation PCKh scores:

  Head    Shoulder    Elbow    Wrist    Hip    Knee    Ankle    Mean
------  ----------  -------  -------  -----  ------  -------  ------
 96.15       94.89    88.14    83.78  87.43   82.19    77.87   87.33
```

Along with the PCKh values, we've provided code to visualize the predicted joints as well in the evaluate_mpii.py

### Train an 8-stack hourglass model

```bash
$ python scripts/train_mpii.py \
    --arch=hg8 \
    --image-path=/path/to/mpii/images \
    --checkpoint=checkpoint/hg8 \
    --epochs=50 \
    --train-batch=6 \
    --test-batch=6 \
    --lr=5e-4 \
    --schedule 150 175 200
    --case=1
    --optim=adam
```

#### To run our modifications - 

Use --case flag and specify case number as shown in table above. In addition, we've added a flag --optim to use Adam or RMSProp for optimizer.

### Hardware used

This code has been run on a single Quadro RTX 6000 GPU with Python 3.8.8, PyTorch 1.8.1 and Torchvision 0.9.1
