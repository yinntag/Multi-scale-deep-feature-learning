# Multi-scale Deep Feature Learning for Human Activity Recognition Using Wearable Sensors
![Image text](https://github.com/yinntag/Multi-scale-deep-feature-learning/blob/main/Model/model.png)
# Abstract
  Deep convolutional neural networks (CNNs) achieve state-of-the-art performance in wearable Human Activity Recognition (HAR), which has become a new research trend in ubiquitous computing scenario. Increasing network depth or width can further improve accuracy. However, in order to obtain the optimal HAR performance on mobile platform, it has to consider a reasonable trade-off between recognition accuracy and resource consumption. Improving the performance of CNNs without increasing memory and computational burden is more beneficial for HAR. In this paper, we first propose a new CNN that uses hierarchical-split (HS) idea for a large variety of HAR tasks, which is able to enhance multi-scale feature representation ability via capturing a wider range of receptive fields of human activities within one feature layer. Experiments conducted on benchmarks demonstrate that the proposed HS module is an impressive alternative to baseline models with similar model complexity, and can achieve higher recognition performance (e.g., 97.28%, 93.75%, 99.02%, and 79.02% classification accuracies) on UCI-HAR, PAMAP2, WISDM, and UNIMIB-SHAR. Extensive ablation studies are performed to evaluate the effect of the variations of receptive fields on classification performance. Finally, we demonstrate that multi-scale receptive fields can help to learn more discriminative features (achieving 94.10% SOTA accuracy) in weakly labeled HAR dataset.
# Requirements
- python 3
- pytorch >= 1.1.0
- torchvision
- numpy 1.21.2
# Usage
You can follow [UCI](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) to prepare the HAR data.

Run `python main.py` to train and test on several HAR datasets. 
# Contributing
We appreciate all contributions. Please do not hesitate to let me know if you have any problems during the reproduction (yinntag@gmail.com).

# Citation
```
@article{tang2022multi,
  title={Multi-scale Deep Feature Learning for Human Activity Recognition Using Wearable Sensors},
  author={Tang, Yin and Zhang, Lei and Min, Fuhong and He, Jun},
  journal={IEEE Transactions on Industrial Electronics},
  year={2022},
  publisher={IEEE}
}
```

