
# DeepCARSKit

*A Deep Learning Based Context-Aware Recommendation Library*

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## History
+ **[CARSKit](https://github.com/irecsys/CARSKit)** was released in 2015, and it was the first open-source library for 
context-aware recommendations. There were no more significant updates in CARSKit since 2019. It was a library built based on Java and [Librec](https://github.com/guoguibing/librec) v1.3. 
There is a version in Python, [CARSKit-API](https://github.com/WagnoLeaoSergio/CARSKit_API), which is a python wrapper of CARSKit.
+ Recommender systems based on deep learning have been well-developed in recent years. The context-aware 
recommendation models based on traditional collaborative filtering (e.g., KNN-based CF, matrix factorization) turned out to 
be out-dated. Therefore, we develop and release [DeepCARSKit](https://github.com/irecsys/DeepCARSKit) which was built upon the [RecBole](https://recbole.io/) v1.0.0 recommendation library.
DeepCARSKit is *a Deep Learning Based Context-Aware Recommendation Library* which can be run with correct setting based on Python and [PyTorch](https://pytorch.org/).


## Feature
+ **Implemented Deep Context-Aware Recommendation Models.** We support the CARS models based on FM and NeuMF. More algorithms will be added.

+ **Multiple Data Splits & Evaluation Options.** We provide evaluations based on both hold-out and N-fold cross validations.

+ **Extensive and Standard Evaluation Protocols.** We rewrite codes in RecBole to adapt the evaluations for context-aware recommendations.
Particularly, item recommendations can be produced for each unique combination of (user and context situation). Relevance and Ranking metrics, 
such as precision, recall, NDCG, MRR, can be calculated by taking context information into consideration.

+ **Other Features.** Other characteristic in DeepCARSKit are inherited from RecBole, suc as GPU accelerations.


## News & Updates
**03/19/2022**: We release DeepCARSKit v1.0.0

## API Documents
You can refer to the API doc of DeepCARSKit produced by Sphinx from [here](https://carskit.github.io/doc/DeepCARSKit/index.html).

We also suggest you to refer to [RecBole API](https://recbole.io/docs/).


## Data Sets & Preparation
A list of available data sets for research on context-aware recommender systems can be found [here](https://github.com/irecsys/CARSKit/tree/master/context-aware_data_sets).
We provide two data sets (i.e., DePaulMovie and TripAdvisor) in the library. You can refer to its data format, such as depaulmovie.inter.

More specifically, you need to prepare a data set looks like this: (use 'float' and 'token' to indicate numerical and nominal variables)

+ user_id:token
+ item_id:token
+ rating:float
+ context variable 1:token
+ context variable 2:token
+ context variable N:token
+ contexts:token => a concatenation of context conditions
+ uc_id:token => a concatenation of user_id and contexts

## Installation
DeepCARSKit works with the following operating systems:

* Linux
* Windows 10
* macOS X

DeepCARSKit requires Python version 3.7 or later, torch version 1.7.0 or later, and RecBole version 1.0.0 or later. 
For more details, you can refer to the list of [requirements](https://github.com/irecsys/DeepCARSKit/blob/main/requirements.txt).

If you want to use DeepCARSKit with GPU,
please ensure that CUDA or cudatoolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

More info about installation from conda and pip will be released later.
Currenly, you can make a git clone of the source codes.

## Quick-Start
With the source code, you can use the provided script for initial usage of our library:

```bash
python run.py
```

This script will run the NeuCMFi model on the DePaulMovie dataset.


### Hyperparameter tuning 
You can tune up the parameters from the configuration file, config.yaml


## Major Releases
| Releases  | Date       |
|-----------|------------|
| v1.0.0    | 03/19/2022 |




## Cite
If you find DeepCARSKit useful for your research or development, please cite the following paper:
(the paper is under review now, more details will be released later)

```
@article{deepcarskit,
    title={DeepCARSKit: A Deep Learning Based Context-Aware Recommendation Library},
    author={Yong Zheng},
    year={2022}
}
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

We welcome collaborations and contributors to the DeepCARSKit. Your names will be listed here.

## Sponsors
The current project was supported by Google Cloud Platform. We are looking for more sponsors to support the development and distribution of this libraray.
If you are interested in sponsorship, please let me know. Our official email is DeepCARSKit [at] gmail [dot] com.

## License
[MIT License](./LICENSE)
