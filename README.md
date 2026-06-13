## Dataset Source

The processed datasets follow the data setting used in PSTUN:

[NIM-NMDC/PSTUN](https://github.com/NIM-NMDC/PSTUN)

Please download the corresponding Chikusei, Xiongan New Area, and WorldView-3 data from the PSTUN repository or follow its data preparation instructions. Put the processed files under the following paths:

```text
data/
  chikusei/
    chikusei_train.h5
    chikusei_val.h5
    chikusei_test.h5
    Chikusei.mat
    chikusei_128_4.mat
  xiongan_new_area_x4/
    xiongan_new_area_train.h5
    xiongan_new_area_val.h5
    xiongan_new_area_test.h5
  WV3/
    train_wv3.h5
    valid_wv3.h5
    reduced_examples/reduced_examples/test_wv3_multiExm1.h5
    full_examples/full_examples/test_wv3_OrigScale_multiExm1.h5
```

For Xiongan New Area SAM-cache construction, also prepare:

```text
DATASET_MIGRATION_BUNDLE/
  xantrain.npy
  chikuseisrf.mat
```


## Training

The main method parameters are enabled by default. The default command trains the Chikusei 4x model:

```bash
python train.py
```

Train Xiongan New Area:

```bash
python train.py --dataset xiongan_new_area
```

Train WV3:

```bash
python train_wv3.py
```

## Testing

Test Chikusei:

```bash
python test.py --weight ./weights/your_experiment/best_by_psnr.pth
```

Test Xiongan New Area:

```bash
python test.py --dataset xiongan_new_area --weight ./weights/your_experiment/best_by_psnr.pth
```

Test WV3 reduced-resolution:

```bash
python test_wv3_rr.py --weight ./weights/your_wv3_experiment/best_by_psnr.pth
```

Test WV3 real-world/original-scale data:

```bash
python test_wv3_real.py --weight ./weights/your_wv3_experiment/best_by_psnr.pth --save_mat
```
