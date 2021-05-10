# Stacked Hourglass Network for Human pose estimation



## Example scripts


### Evaluation on the MPII validation set

Here's a quick example of evaluating the pretrained 2-stack hourglass model on the MPII Human
Pose validation set.

```bash
$ python scripts/evaluate_mpii.py --arch=hg2 --image-path=/path/to/mpii/images
```

Output:

```
Final validation PCKh scores:

  Head    Shoulder    Elbow    Wrist    Hip    Knee    Ankle    Mean
------  ----------  -------  -------  -----  ------  -------  ------
 96.15       94.89    88.14    83.78  87.43   82.19    77.87   87.33
```


### Train an 8-stack hourglass model

```bash
$ python scripts/train_mpii.py \
    --arch=hg8 \
    --image-path=/path/to/mpii/images \
    --checkpoint=checkpoint/hg8 \
    --epochs=220 \
    --train-batch=6 \
    --test-batch=6 \
    --lr=5e-4 \
    --schedule 150 175 200
```

### Hardware used to test

This code has been run on a single Quadro RTX 6000 GPU with Python 3.8.8, PyTorch 1.8.1 and Torchvision 0.9.1
