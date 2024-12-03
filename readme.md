# Scale invariant Laplace approximations


This project is based on:

[G-SGD: Optimizing ReLU Neural Networks in its Positively Scale-Invariant Space](https://arxiv.org/abs/1802.03713)

[G-SGD Github repository](https://github.com/MSRA-COLT-Group/gsgd)

[Laplace Redux -- Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806)



# How to run

`python3 main.py --batch_size 64 --learning_rate 0.01 --optimizer all --epochs 200 --dataset mnist`



### Parameters

| Parameter | Values                           | Description                                                  |
| --------- | -------------------------------- | ------------------------------------------------------------ |
| optimizer | sgd<br />adam<br />gsgd<br />all | You can run the code with `all` to compare the different optimizers.<br />In `optimizers/gsgd` you can find the gsgd implementation |
| dataset   | mnist<br />cifar10<br />cifar100 | `Mnist` and `Cifar10` use the same model as in the original G-SGD paper |

The model is currently also selected with the dataset parameter.



# Code structure

The code saves the model weights, test and validation results alongside a plot in output/

The results are currently being generated.
