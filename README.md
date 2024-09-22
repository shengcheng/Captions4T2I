# Precision or Recall? An Analysis of Image Captions for Training Text-to-Image Generation Model

This repository contains the codebase for [paper](): Precision or Recall? An Analysis of Image Captions for Training Text-to-Image Generation Model

## Dataset Construction
Download the raw captions from the DCI dataset using the following [link](https://github.com/facebookresearch/DCI/blob/3d95af66c918f0cc24f115c5eeb0d2f66be30872/dataset/densely_captioned_images/dataset/scripts/download.py#L16-L29).

To pre-process these captions, run the following command:
```
python dci_data.py
```
To generate captions with varying levels of precision and recall, please run:
```
python generate_captions.py
```

## Training
For efficient training, first pre-process the captions using:
```
python pre_process.py
```
To train the constructed dataset on the pixartAlpha model, use the following command:
```
python train_pixart_dci.py
```

## Acknowledgement
We would like to acknoweldge the data from [DCI](https://github.com/facebookresearch/DCI) and T2I model [PixartAlpha](https://github.com/PixArt-alpha/PixArt-alpha)  for their contributions.. 

