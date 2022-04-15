## S4A Models
#### Institute of Astronomy, Astrophysics, Space Applications and Remote Sensing (IAASARS), National Observatory of Athens (NOA)

**Contributors:** [Sykas D.](https://github.com/dimsyk), [Zografakis D.](https://github.com/dimzog), [Sdraka M.](https://github.com/paren8esis)

**This repository contains the models and training scripts for reproducing the experiments presented in:**\
[A Sentinel-2 multi-year, multi-country benchmark dataset for crop classification and segmentation with deep learning](https://ieeexplore.ieee.org/document/9749916).

### Description

Based on the Sen4AgriNet dataset, we produce two distinct sub-datasets of Sentinel-2 L1C images for experimentation:
- **Patches Assembled Dataset (PAD)**: all Sentinel-2 images expanding over 2 years (2019, 2020) and 2 regions (France, Catalonia) with pixel-wise labels for crop classification (overall 168 classes).
- **Object Aggregated Dataset (OAD)**: on top of PAD, the mean and std of each parcel is computed for each Sentinel-2 observation and used as input data. A single label for each parcel is used.

The above sub-datasets were downsized in order to be used for the experiments in this repository. Specifically, 5000 patches with 60-20-20 (train-val-test) split were sampled from each sub-dataset and the 11 most frequent classes were kept (*wheat*, *maize*, *sorghum*, *barley*, *rye*, *oats*, *grapes*, *rapeseed*, *sunflower*, *potatoes*, *peas*). Three different scenarios were explored:

Scenario | Train | Test
--|---|---
1  | Catalonia (2019, 2020), France (2019) | Catalonia (2019, 2020), France (2019)
2  | Catalonia (2019, 2020) | France (2019)
3  | France (2019) | Catalonia (2020)

The input of the PAD models is the median of each month of observations from April through September. The OAD models take as input the aggregated statistics of these observations.

### Requirements

This repository was tested on:
* Python 3.8
* CUDA 11.4
* PyTorch 1.11
* PyTorch Lightning 1.6

Check `requirements.txt` for other essential modules.

### Available models

For PAD:
1. [ConvLSTM](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)
2. [ConvSTAR](https://www.sciencedirect.com/science/article/pii/S0034425721003230)
3. [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
4. [TempCNN](https://www.mdpi.com/2072-4292/11/5/523)

For OAD:
1. [TempCNN](https://www.mdpi.com/2072-4292/11/5/523)
2. [LSTM](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)
3. [Transformer](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

### Instructions

#### Folder structure

```
S4A-models\
    L dataset\
        L netcdf\
        L oad\
    L coco_files\
    L logs\
        L medians
    L model\
    L utils\
        L settings\
            L mappings\
```

#### COCO files

In the `coco_files/` folder are the COCO files required for training, validating and testing the models.

#### NetCDF4 files

In the `dataset/netcdf/` folder you should place the netCDF4 files.

#### OAD files

In the `dataset/oad/` folder you should place the exported files containing the OAD statistics.

#### Configuration

* The main configuration file is `utils/settings/config.py`.
* Custom Taxonomy Mapping is located at `utils/settings/mappings/mappings_{cat, fr}.py`.

Every script inherits settings from the aforementioned files.

#### Essential scripts

- `coco_data_split.py`: Uses the nectCDF4 data to produce three COCO files for training, validation and testing.
- `export_medians_multi.py`: Uses the netCDF4 data and the COCO files to compute the median image per month and export them to the disk.
- `compute_class_weights.py`: Computes the class weights based on the exported medians, to account for class imbalance.
- `object-based-csv.py`: Uses the netCDF4 data to compute the statistics required for OAD.
- `pad_experiments.py`: The main script for training/testing the PAD models.
- `oad_experiments.py`: The main script for training/testing the OAD models.
- `visualize_predictions.py`: Produces a visualization of the ground truth and the prediction of a given model for a given image. Only relevant for PAD models.

#### Using the repo

**Preparation**
1. Run `export_medians_multi.py` to precompute the medians needed for training, validation and testing.
2. If you don't want to use the given COCO files, then export your own using the `coco_data_split.py` script.
3. Uncomment the precomputed class weights in the corresponding section of the configuration file depending on the scenario you are using (or compute your own).
4. Especially for OAD, run `object-based-csv.py` to export the statistics needed for the experiments.

**For PAD:**
1. Run `pad_experiments.py` with the appropriate arguments. Example:
   ```
   python pad_experiments.py --train --model convlstm --parcel_loss --weighted_loss --root_path_coco <coco_folder_path> --prefix_coco <coco_file_prefix> --prefix <run_prefix> --num_epochs 10 --batch_size 32 --bands B02 B03 B04 B08 --saved_medians --img_size 61 61 --requires_norm --num_workers 16 --num_gpus 1 --fixed_window
   ```
   The above command is for training the **ConvLSTM** model using the **weighted parcel loss** described in the associated publication. Training will continue for **10 epochs** with **batch size 32**, using the Sentinel-2 **bands Blue (B02), Green (B03), Red (B04) and NIR (B08)**. The **input image size is 61x61**, the **precomputed medians are used** to speed up training and all input data are **normalized**. Finally, a **fixed window** is used containing months 4 (April) through 9 (September). Please use the `--help` argument to find information on all available parameters.
2. Optionally, after training run `visualize_predictions.py` to visualize the image, ground truth and prediction for a specific model and image.

**For OAD:**
1. Run `oad_experiments.py` with the appropriate arguments. Example:
   ```
   python oad_experiments.py --train --model transformer --prefix <run_prefix> --file <oad_file_name> --num_epochs 10 --batch_size 32 --num_workers 16 --num_gpus 1 --hidden_size 1024 --num_layers 3
   ```
   The above command is for training the **Transformer** model. Training will continue for **10 epochs** with **batch size 32**, using given **file containing the OAD statistics**. The **hidden size is 1024** and **three layers** are used for the model. Please use the `--help` argument to find information on all available parameters.

### Reported results

The results reported on the given COCO files are presented in the following tables.

#### PAD

Scenario | Model | Acc. W. (%) | F1 W. (%) | Precision W. (%)  
--|---|---|---|--
1  | U-Net | 93.70  | 82.61 | 86.64
1  | ConvLSTM  | **94.72**  | **85.18** | **86.86**
1  | ConvSTAR | 92.78 | 80.38 | 83.33
2  | U-Net | **83.12** | **57.85** | **61.57**
2  | ConvLSTM | 82.53 | 56.56 | 60.57
2  | ConvSTAR | 79.52 | 52.15 | 58.98
3  | U-Net | **72.11** | **43.54** | **68.42**  
3  | ConvLSTM | 69.86 | 40.47 | 66.17
3  | ConvSTAR | 69.07 | 34.45 | 67.43

#### OAD

Scenario | Model | Acc. W. (%) | F1 W. (%) | Precision W. (%)  
--|---|---|---|--
1  | LSTM | 88.52 | 88.03 | 87.85
1  | Transformer | 88.36  | 88.10 | 87.90
1  | TempCNN | **90.08** | **89.97** | **90.01**
2  | LSTM | **91.55** | **91.34** | **91.31**
2  | Transformer | 39.17 | 31.45 | 58.52
2  | TempCNN | 36.90 | 30.14 | 60.71
3  | LSTM | **60.60** | **63.96** | **70.55**  
3  | Transformer | 51.21 | 56.71 | 67.76
3  | TempCNN | 52.32 | 57.38 | 68.35

### Citation

If you use our work, please cite:

```
@ARTICLE{
  9749916,
  author={Sykas, Dimitrios and Sdraka, Maria and Zografakis, Dimitrios and Papoutsis, Ioannis},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  title={A Sentinel-2 multi-year, multi-country benchmark dataset for crop classification and segmentation with deep learning},
  year={2022},
  doi={10.1109/JSTARS.2022.3164771}
}
```
