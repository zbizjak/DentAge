# DentAge

DentAge is a project aimed at estimating the age of dental patients using machine learning techniques. This repository contains the code and resources needed to train and evaluate models for dental age estimation, based on recent publication:
[DentAge: Deep learning for automated age prediction using panoramic dental X-ray images](https://onlinelibrary.wiley.com/doi/full/10.1111/1556-4029.15629)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Cite](#Citation)

## Introduction

The goal of this project is to develop a reliable method for estimating the age of dental patients based on their dental images. This can be useful in various fields such as forensic science, orthodontics, and pediatric dentistry.

## Installation

To get started with DentAge, clone the repository and build docker image from dockerfile:

```bash
git clone https://github.com/zbizjak/DentAge.git
cd DentAge
DOCKER_BUILDKIT=1 docker build -t pytorch_dentage . -f Dockerfile
```

## Usage

To train the model, use the following command:

```bash
docker run --name DentAge --gpus all -it --rm -v $(pwd):/app pytorch_dentage python3 train.py
```

To evaluate on test case, download model from release and run:

```bash
docker run --name DentAge --gpus all -it --rm -v $(pwd):/app pytorch_dentage python3 predict_example.py
```


## Contributing

We welcome contributions to DentAge! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License.


## Citation
If you use this code or dataset in your research, please cite our paper [DentAge: Deep learning for automated age prediction using panoramic dental X-ray images](https://onlinelibrary.wiley.com/doi/full/10.1111/1556-4029.15629):

```
@article{bizjakdentage,
  title={DentAge: Deep learning for automated age prediction using panoramic dental X-ray images},
  author={Bizjak, {\v{Z}}iga and Robi{\v{c}}, Tina},
  journal={Journal of Forensic Sciences},
  publisher={Wiley Online Library}
}
```