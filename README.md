Cross-study analysis (CSA) with cell-line drug sensitivity datasets.

## Generate data
Generate data per drug sensitivity study (e.g., cancer and drug features, response values)

```
python src/build_dfs_july2020.py
```

The data is saved in `ml.dfs/July2020/data.STUDY_NAME` (e.g., data.ccle, data.ctrp, etc). Each dir contains multiple files. For example, in data.GDSC1:
* `rsp_gdsc1.csv`: drug response samples with drug ids (DrugID), cell ids (CancID), and multiple measures of drug response. For drug response, we use the `AUC` (area under the dose-response curve).
* `ge_gdsc1.csv`: gene expression. The first column is CancID, and the remaining columns are gene names (GeneBank) prefixed with `ge_`.
* `mrd_gdsc1.csv`: mordred descriptors. The first column is DrugID, and the remaining columns are mordred descriptors prefixed with `mordred_`.
* `fps_gdsc1.csv`: ECFP2 morgan fingerprints (radius=2, nBits=512). The first column is DrugID, and the remaining columns are fingerprints prefixed with `mordred_` or `ecfp2_`.
* `splits`: This folder contains multiple sets of train/test split ids. Each split comes with a pair of files:
    * Ids for training: split_{split_number}_tr_id
    * Ids for testing: split_{split_number}_te_id
The ML model should be able to train using training samples (the split_{split_number}_tr_id) and test using test samples (the split_{split_number}_te_id).

## Preprocessing, training, inference
The ML model should be able to take data from data.STUDY_NAME, pre-process to conform the model's training/test API, train the model, and save predictions. These steps are model-depend. All models should save raw predictions while following the format and naming convention as required by the CSA API. Check `train_all.py` for doing this with LightGBM.

```
python train_all.py
```

__Preprocessing__. Take data.STUDY_NAME and generate data structures to conform the model's train/test API. For example:
* LightGBM can take csv files
* GraphDRP requires PyTorch data loaders for train, val, and test sets

__Training__. Train the model using split ids (each split produces a trained model). The training dataset is the __source__ dataset. For example:
train the model using ids of a specific split
                 e.g.: train model the with ccle split 0 train ids

__Inference__. Using the trained models for each split, run inferene with all datasets. The inference datasets are the `target` datasets. When `source` and `target` are the same, use test ids for inference. Alternatively, when `source` and `target` are different, run inference for all the available samples.

A single file of model predictions should be created for a given source dataset, target dataset, and split as follows: SOURCE_TARGET_split_#.csv
The naming convetion for 

Thus, if `source` is data.ccle, there are 5 `target` datasets (data.ccle, data.ctrp, data.gcsi, data.gdsc1, data.gdsc2) and 10 data splits, the model should produce raw predictions in the following files:
* target is ccle (test ids)     ccle_ccle_split_0.csv, ..., ccle_ccle_split_9.csv
* target is ctrp (all samples)  ccle_ctrp_split_0.csv, ..., ccle_ctrp_split_9.csv
* target is gcsi (all samples)  ccle_gcsi_split_0.csv, ..., ccle_gcsi_split_9.csv
* target is gdsc1 (all samples) ccle_gdsc1_split_0.csv, ..., ccle_gdsc1_split_9.csv
* target is gdsc2 (all samples) ccle_gdsc2_split_0.csv, ..., ccle_gdsc2_split_9.csv

Each such file must contain the following 4 columns: `DrugID`, `CancID`, `True`, and `Pred`. The DrugID and CancID are drug and cell line ids (used in data.STUDY_NAME files). The True and Pred are the ground truth and predictions, respetively. 

All these files should be stored in a single folder `results.MODEL_NAME` (e.g., results.LGBM, results.IGTD, results.UNO).

## CSA API
The CSA API will take raw model predictions from results.MODEL_NAME and generate CSA tables for the model.
