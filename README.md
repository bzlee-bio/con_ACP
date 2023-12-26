# Contrastive learning for enhancing feature extraction in anticancer peptides

A deep learning model designed to screen ACPs using peptide sequences only. A contrastive learning technique was applied to enhance model performance, yielding better results than a model trained solely on binary classification loss. Furthermore, two independent encoders were employed as a replacement for data augmentation, a technique commonly used in contrastive learning.

## Dependencies
pytorch>=2.0.1
numpy>=1.25.2

## Datasets
Datasets for model training was obtained from [ACPred-LAF](https://github.com/TearsWaiting/ACPred-LAF).
There were five benchmark datasets were used for model training.
- ACP-Mixed-80
- ACP2.0 main
- ACP2.0 alternative
- ACP500+ACP164
- ACP500+ACP2710
- LEE+Independent

Detailed information can be found [here](https://academic.oup.com/bioinformatics/article/37/24/4684/6330613).

## Model training
```
python train.py --model_info {model_info} --batch_size {batch_size} --dropout_rate {dropout_rate}
                --lr {learning_rate} --epoch {maximum_training_epochs} --dataset {bechmark_dataset}
                --alpha {alpha} --beta {beta} --temp {temperature} --gpu {gpu_number}
```
- <b>model_info</b>: Encoder architecture meta file is in `./model/model_params`. Among these encoder architectures, select one encoder architecture for model training. e.g. `--model_info ./model/model_params/cnn1.json`.
- <b>batch_size</b>: Batch size for model training
- <b>dropout_rate</b>: Dropout rate for model training
- <b>learning_rate</b>: Learning rate for model training
- <b>maximum_training_epochs</b>: Maximum epochs for model training
- <b>benchmark_dataset</b>: Among six benchmark datasets, select one dataset for model training.
  - <b>Options</b>: ACP_Mixed-80 for ACP-Mixed-80
- <b>alpha</b>: Control the balance between cross-entropy and contrastive loss components. (0.0 ~ 1.0)
- <b>beta</b>: Control the balance between two cross-entropy losses (cross-entropy loss 1 and 2)
- <b>temperature</b>: Temperature parameter in contrastive loss
- <b>gpu</b>: Number of GPU that will be used for model training. GPU number can be found by the command of `nvidia-smi`. For using CPU, provides `-1`.