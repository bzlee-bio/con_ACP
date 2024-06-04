# Contrastive learning for enhancing feature extraction in anticancer peptides

A deep learning model designed to screen anticancer peptides (ACPs) using peptide sequences only. A contrastive learning technique was applied to enhance model performance, yielding better results than a model trained solely on binary classification loss. Furthermore, two independent encoders were employed as a replacement for data augmentation, a technique commonly used in contrastive learning.

## Dependencies
- pytorch>=2.0.1
- numpy>=1.25.2
- biopython

## Datasets
Datasets for model training were obtained from [ACPred-LAF](https://github.com/TearsWaiting/ACPred-LAF).
Six benchmark datasets were used for model training:
- ACP-Mixed-80
- ACP2.0 main
- ACP2.0 alternative
- ACP500+ACP164
- ACP500+ACP2710
- LEE+Independent

For more detailed information, refer to this [research article](https://academic.oup.com/bioinformatics/article/37/24/4684/6330613).

## Inference

To predict ACPs using only peptide sequences, prepare your peptide sequence list in the FASTA format. For more detailed information about the FASTA format, refer to [this link](https://en.wikipedia.org/wiki/FASTA_format).

### Method 1: Command-Line Inference
Use the following command to run the inference:

```
python inf.py --batch_size {batch_size} --model_type {model_type}
              --device {device} --output {output_file}
```
- <b>batch_size</b>: The batch size used during inference
- <b>model_type</b>: Specifies the type of optimized model. There are six optimized models available for predicting ACPs, each trained on one of six benchmark datasets. The default recommended option is ACP-Mixed-80.
  - <b>Options</b>
    - `ACP_Mixed_80`: The optimized model that was trained using the ACP-Mixed-80 benchmark dataset.
    - `ACP2_main`: The optimized model that was trained using the ACP2.0 main benchmark dataset.
    - `ACP2_alter`: The optimized model that was trained using the ACP2.0 alternative benchmark dataset.
    - `ACP500_ACP164`: The optimized model that was trained using the ACP500+ACP164 benchmark dataset.
    - `ACP500_ACP2710`: The optimized model that was trained using the ACP500+ACP2710 benchmark dataset.
    - `LEE_Indep`: The optimized model that was trained using the LEE+Independent benchmark dataset.
- <b>device</b>: The device used for predicting ACPs
  - <b>Options</b>
    - `cpu`
    - `gpu`
- <b>output_file</b>: The file where prediction results will be saved.

### Method 2: Using the acppred Python Package
Alternatively, you can utilize the acppred Python package for predictions

- Install acppred: First, install the acppred package using pip:
```bash
pip install acppred
```

- Predict Using acppred: Utilize the following Python script to perform ACP predictions:
```python
import acppred as ap

ap.predict(fasta_file, model_type="ACP_Mixed_80", device="cpu", batch_size=64)
```
- fasta_file: The path to the FASTA file containing the peptide sequences you want to analyze.
- model_type: Specifies the machine learning model to use for the prediction. 
- device: Indicates whether to use the CPU ("cpu") or GPU ("gpu") for computation.
- batch_size: Determines the number of sequences to process simultaneously. A larger batch size can expedite the prediction process but will require more memory. Adjust this parameter based on your system's capabilities and the size of your dataset.

This script facilitates ACP prediction by integrating the acppred package, allowing you to specify the model type, computing device, and batch size.




Note: Due to variability in the maximum peptide sequence length across each benchmark dataset, there are restrictions on the maximum input peptide sequence length for each model type.
|Model Type|Maximum Number of Amino Acid Residues|
|---|:---:|
|ACP2_main|50|
|ACP2_alter|50|
|LEE_Indep|95|
|ACP500_ACP164|206|
|ACP500_ACP2710|206|
|ACP_Mixed_80|207|



## Model training
Use the following command to start model training:
```
python train.py --model_info {model_info} --batch_size {batch_size} --dropout_rate {dropout_rate}
                --lr {learning_rate} --epoch {maximum_training_epochs} --dataset {bechmark_dataset}
                --alpha {alpha} --beta {beta} --temp {temperature} --gpu {gpu_number}
```
- <b>model_info</b>: Choose an encoder architecture from the `./model/model_params` directory for model training. For example, `--model_info ./model/model_params/cnn1.json`.
- <b>batch_size</b>: Batch size used during model training
- <b>dropout_rate</b>: Dropout rate applied during model training
- <b>learning_rate</b>: Learning rate set for model training.
- <b>maximum_training_epochs</b>: Maximum number of training epochs.
- <b>benchmark_dataset</b>: Select one dataset from the six available benchmark datasets for model training.
  - <b>Options</b>
    - `ACP_Mixed-80`: ACP-Mixed-80 dataset
    - `ACP2_main`: ACP2.0 main dataset
    - `ACP2_alter`: ACP2.0 alternative dataset
    - `ACP500_ACP164`: ACP500+ACP164 dataset
    - `ACP500_ACP2710`: ACP500+ACP2710 dataset
    - `LEE_Indep`: LEE+Independent dataset
 
- <b>alpha</b>: Adjusts the balance between cross-entropy and contrastive loss components. Range: 0.0 to 1.0.
- <b>beta</b>: Balances the two types of cross-entropy losses (cross-entropy loss 1 and 2).
  - <b>Options</b>
    - `0`: Only cross-entroly loss 1 is used for model training.
    - `0.5`: Both cross-entropy loss 1 and 2 are used for model training.
    - `1`: Only cross-entroly loss 2 is used for model training.
- <b>temperature</b>: Temperature parameter in contrastive loss calculation.
- <b>gpu</b>: GPU number to be used for model training, as identified by the `nvidia-smi`` command. Use `-1`` for CPU training.

## Reference
Byungjo Lee, Dongkwan Shin, Contrastive learning for enhancing feature extraction in anticancer peptides, Briefings in Bioinformatics, Volume 25, Issue 3, May 2024, bbae220, https://doi.org/10.1093/bib/bbae220
