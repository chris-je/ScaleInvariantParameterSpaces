# Scale invariant parameter spaces

This project aims to abstract and test different implementations of a scale-invariant parameter spaces in Pytorch.

Running the main script allows to compare different optimizers, which we can either run on the default parameter space, the G-space or the Fixed space.



**This project is based on:**

[G-SGD: Optimizing ReLU Neural Networks in its Positively Scale-Invariant Space](https://arxiv.org/abs/1802.03713)

[G-SGD Github repository](https://github.com/MSRA-COLT-Group/gsgd)



**Parameter spaces:**

The two implementation in particular are:

- G-space, based on the G-SGD method. Computes a parameter space out of paths which span the network from input to output.
- Fixed weights, fixes some weights in place by setting gradients to 0



# How to run

`python3 main.py --learning_rate 0.01 --optimizer sgd gsgd --dataset mnist`



### Parameters

| Parameter     | Values                                                       | Default | Description                                                  |
| ------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| batch_size    | \<int\>                                                      | 64      |                                                              |
| learning_rate | \<float\>                                                    | 0.01    | For comparison, you can add multiple learning rates separated with a space |
| optimizer     | sgd<br />adam<br />gsgd<br />gadam<br />fsgd<br />fadam<br /> | all     | You can use multiple optimizers separated with a space. Optmimizers starting with a g work on the G-space and the ones starting with an f work on the fixed weight space |
| dataset       | mnist<br />cifar10<br />fashionmnist                         |         | You can use multiple datasets separated with a space. Each will compute it's own results |
| epochs        | \<int\>                                                      | 50      | Specify the number of training epochs                        |
| runs          | \<int\>                                                      | 1       | Run model multiple times with the same settings to compute an average |
| save_weights  | -                                                            | false   | When using this flag, the weights will be saved              |
| output        | \<string\>                                                   |         | specify a specific output folder                             |



## Function calls to transform into new parameter space

In the following example, `optimizer` will run Adam on the G-space:

`model_parameters = list(model.parameters())`

`optimizer_class = optim.Adam`

`learning_rate = 0.05`

`optimizer = GOptmizer(model_parameters, optimizer_class, learning_rate)`



The following example will fix the model's parameters:

`FixModel(model, fixed_factor)`

Fixed factor can either be a boolean or a foat. When it is a boolean, it will either fix the weights without changing them (False) or setting them to 1 (True). For any foat, it will set the fxed weights to the given number. Using regularizers on the fixed model may still unintentionally update the value of the fixed
weights.



## Folder structure

- The code saves the model weights, test and validation results alongside a plot in `results/`

- The implementations of the parameter spaces are in `parameterSpaces/`
- `experiments` contains a series of Plots visualizing the parameter spaces
