# MMOE
Implementation of paper [《Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts》](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts)

# Dataset
census data set from [PaddleRec](https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets/census)

# Run
* create `data` and `data/tfrecords` folders
* download and move `train_data.csv` and `test_data.csv` to `data` folder
* Run with default config: `python main.py`

# Key points
* batch norm for census data

# TODOs
* try tencent video data set
* MMOE with attention
* grad norm
