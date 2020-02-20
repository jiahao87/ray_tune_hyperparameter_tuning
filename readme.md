## Overview
This repository contains the code template for hyperparameter tuning using Ray Tune.

Ray Tune is a Python library for hyperparameter tuning at any scale, allowing us to easily perform multi-node distributed computing to evaluate various hyperparameter configurations at the same time.

![Ray Tune logo](https://ray.readthedocs.io/en/latest/_images/tune.png)

Guidelines for using code:
1. Setup Google Cloud Authentication
* Create service account and download JSON file that contains your key
* Set environment variable to point to the directory of the JSON file downloaded
* Reference: https://cloud.google.com/docs/authentication/getting-started

2. Enable the following APIs
* Cloud resource manager API
* Identity and Access Management (IAM) API
* Compute Engine API

3. Copy paste project ID to project_id in cluster_config_cpu.yml config file
4. Launch your cluster by running in terminal:
```
ray up -y cluster_config_cpu.yml
```
5. Start hyperparameter tuning trials by executing in terminal:
```
ray submit cluster_config_cpu.yml tune_cifar10.py
# To trial run scripts, add argument smoke-test
# ray submit cluster_config_cpu.yml tune_cifar10.py --args="--smoke-test"
```
