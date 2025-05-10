# EvoNet
This project implements the Evolutionary State Graph Neural Network proposed in [1], which is a GNN-based state-recognition and graph based method for time-series event prediction. This work is done as part of IE 506 Course Project which is reproduction of results of Paper 'Time Series Event Prediction using EvoNet' along with improvements, which is compatible with latest libraries, and the one shot results after improvements obtained after that are as follows:

Package -> Previously Compatible with -> Now Compatible with
Python -> 3.6.2 ->  3.12.3
Tensorflow ->  1.1.0 ->  2.18.0
Numpy ->  1.11.0 -> 2.0.2
Scikit-learn ->  0.19.1 -> 1.6.1
xgboost -> 0.80 -> 2.1.4
tslearn -> 0.1.24 -> 0.6.3

Achieved Improvements on Dataset DJIA30:
Metric -> Result in Paper -> Result after Improvement
Accuracy -> 0.7645 -> 0.6814
Precision -> 0.4711 -> 0.3689
Recall -> 0.3359 -> 0.5753
F1 -> 0.3922 -> 0.4495
AUC -> 0.6471 -> 0.6747

## Compatibility

Code is compatible with tensorflow version 2.18.0 with backward compatibility using tf.compat.v1 and Pyhton 3.12.3.

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

### Input Format 

An example data format is given where data is stored as a list containing 4 dimensional tensors such as
 
`[number of samples × segment number × segment length × dimension of observation]`


### Configuration
We can use `./model_core/config.py` to set the parameters of model.

```
class ModelParam(object):
    # basic
    model_save_path = "./model"
    n_jobs = os.cpu_count()

    # dataset
    data_path = './data'
    data_name = 'webtraffic'
    his_len = 15
    segment_len = 24
    segment_dim = 2
    n_event = 2
    norm = True

    # state recognition
    n_state = 30
    covariance_type = 'diag'

    # model
    graph_dim = 256
    node_dim = 96
    learning_rate = 0.001
    batch_size = 1000
    id_gpu = '0'
    pos_weight = 1.0
```


### Main Script

```
python run.py -h

usage: run.py [-h] [-d {djia30, webtraffic}] [-g GPU]

optional arguments:
  -h, --help            show this help message and exit
  -d {djia30,webtraffic}, --dataset {djia30,webtraffic} select the dataset
  -g GPU, --gpu GPU     target gpu id
```

### Demo Test

```
For a quick verification of the results on a subset of dataset from the saved model:
Run the one block script in demo.ipynb.
The written script uses the saved trained model checkpoints of DJIA 30 dataset, to calculate the evaluation metrics on the subset of test data. 
The sample subset size can be changed using the variable name: DEMO_SAMPLES

```

## Reference

[1] Wenjie, H; Yang, Y; Ziqiang, C; Carl, Y and Xiang, R, 2021, Time-Series Event Prediction with Evolutionary State Graph, In WSDM, 2021 <br>
[2] Wenjie, H; Yang, Y; Ziqiang, C; Carl, Y and Xiang, R, 2021, https://github.com/zjunet/EvoNet <br>
[3] Lin, Girshick, R., He, K., Doll´ar, P. 2017. Focal Loss for Dense Object Detection. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2980–2988.
