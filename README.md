# CekaPlus
Crowd Environment and its Knowledge Analysis - Plus

The first edition of Ceka is available [here](http://ceka.sourceforge.net).

## Identifiers for Instances, Workers, Labels and Class Values
All internal identifiers for instances, workers, labels and class values start from one.\
Zero means an empty annotation.

## Input Files
1. **.resp** file: the file containing the responses of crowdsourced workers to the questions
- format for *single-label* `worker_id` `instance_id` `label_value`
- format for *multi-label* `worker_id` `instance_id` `label_id` `label_value`
2. **.gold** file: the file containing the ground truth for performance evaluation
- format for *single-label* `instance_id` `label_value`
- format for *multi-label*  `instance_id` `label_id` `label_value`

## Algorithms
1. **Multi-Class Multi-Label Independent (MCMLI)**
2. **Multi-Class Multi-Label Dependent (MCMLD)**

## Contact
Jing Zhang (Associate Professor at Nanjing University of Science and Technology, China)\
Email: `jzhang@njust.edu.cn`
