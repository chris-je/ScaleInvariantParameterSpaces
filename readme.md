# Scale invariant Laplace approximations

This project aims to abstract and test the [G-Space](https://arxiv.org/abs/1802.03713) for the use with laplace approximations.

Running the main script will create plots and weights allowing the comparison of different learning rates and optimizers.



**This project is based on:**

[G-SGD: Optimizing ReLU Neural Networks in its Positively Scale-Invariant Space](https://arxiv.org/abs/1802.03713)

[G-SGD Github repository](https://github.com/MSRA-COLT-Group/gsgd)

[Laplace Redux -- Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806)



# How to run

`python3 main.py --learning_rate 0.01 --optimizer sgd gsgd --dataset mnist`



### Parameters

| Parameter     | Values                                             | Default | Description                                                  |
| ------------- | -------------------------------------------------- | ------- | ------------------------------------------------------------ |
| batch_size    |                                                    | 64      |                                                              |
| learning_rate |                                                    | 0.01    | For comparison, you can add multiple learning rates separated with a space |
| optimizer     | sgd<br />adam<br />gsgd<br />                      | all     | You can use multiple optimizers separated with a space.<br />You can find the gsgd implementation in `optimizers/gsgd` |
| dataset       | mnist<br />cifar10<br />cifar100<br />fashionmnist |         | You can use multiple datasets separated with a space.<br />`mnist` and `cifar10` use the same model as in the original G-SGD paper |
| epochs        |                                                    | 50      | How often to iterate through the training data during training |
| runs          |                                                    | 1       | Run model multiple times with the same settings to compute an average |
| save_weights  | -                                                  | false   | When using this flag, the weights will be saved              |

The model is currently also selected with the dataset parameter.



# Code structure

The code saves the model weights, test and validation results alongside a plot in output/

The results are currently being generated.
