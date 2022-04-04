## S4A Models Main Repository
### Institute of Astronomy, Astrophysics, Space Applications and Remote Sensing (IAASARS)
#### National Observatory of Athens (NOA)

Contributors: [Sykas D.](https://github.com/dimsyk), [Zografakis D.](https://github.com/dimzog), [Sdraka M.](https://github.com/paren8esis)


This repository contains the models and training scripts for reproducing the experiments presented in [add publication] .  

#### Requirements

This repository was tested on:
* Python 3.8
* CUDA 11.2
* PyTorch 1.8.1
* PyTorch Lightning 1.2.1

Check `requirements.txt` for other essential modules.

#### Changing Defaults

* Configuration file `utils/settings/config.py`.
* Custom Taxonomy Mapping at `utils/settings/mappings/mappings_{cat, fr}.py`.

Every script inherits settings from the aforementioned files.

#### Essential scripts

- `coco_data_split.py`: Uses the nectCDF4 data and the annotations to produce three COCO files for training, validation and testing.
- `export_medians_multi.py`: Uses the netCDF4 data and the COCO files to compute the median image per month.
- `compute_class_weights.py`: Computes the class weights based on the exported medians, to account for class imbalance.
- `object-based-csv.py`: Uses the netCDF4 data to compute the statistics required for OAD.
- `pad_experiments.py`: The main script for training/testing the PAD models.
- `oad_experiments.py`: The main script for training/testing the OAD models.
- `visualize_predictions.py`: Produces a visualization of the ground truth and the prediction of a given model for a given image.

#### Available models

For PAD:
1. [ConvLSTM](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)
2. [ConvSTAR](https://www.sciencedirect.com/science/article/pii/S0034425721003230)
3. [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
4. [TempCNN](https://www.mdpi.com/2072-4292/11/5/523)

For OAD:
1. [TempCNN](https://www.mdpi.com/2072-4292/11/5/523)
2. [LSTM](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)
3. [Transformer](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

#### Instructions

For PAD:
1. Run `export_medians_multi.py` to precompute the medians needed for training, validation and testing.
2. Run `pad_experiments.py` with the appropriate arguments. Example:
   ```
   python pad_experiments.py --train --model convlstm --parcel_loss --weighted_loss --root_path_coco <coco_folder_path> --prefix_coco <coco_file_prefix> --prefix <run_prefix> --num_epochs 10 --batch_size 32 --bands B02 B03 B04 B08 --saved_medians --img_size 61 61 --requires_norm --num_workers 16 --num_gpus 1 --window_len 12
   ```
   The above command is for training the **ConvLSTM** model using the **weighted parcel loss** described in the associated publication. Training will continue for **10 epochs** with **batch size 32**, using the Sentinel-2 **bands Blue (B02), Green (B03), Red (B04) and NIR (B08)**. The **input image size is 61x61**, the **precomputed medians are used** to speed up training and all input data are **normalized**. The **window length is 12**, including all months. Please use the `--help` argument to find information on all available parameters.
3. Optionally, run `visualize_predictions.py` to visualize the image, ground truth and prediction for a specific model and image.

For OAD:
1. Run `object-based-csv.py` to export the statistics needed for OAD.
2. Run `oad_experiments.py` with the appropriate arguments. Example:
   ```
   python oad_experiments.py --train --model transformer --prefix <run_prefix> --file <oad_file_name> --num_epochs 10 --batch_size 32 --num_workers 16 --num_gpus 1 --hidden_size 1024 --num_layers 3
   ```
   The above command is for training the **Transformer** model. Training will continue for **10 epochs** with **batch size 32**, using given **file containing the OAD statistics**. The **hidden size is 1024** and **three layers** are used for the model. Please use the `--help` argument to find information on all available parameters.
