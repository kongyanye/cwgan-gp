## Conditional WGAN-GP (cWGAN-GP) in Keras ##

**Conditional version of WGAN-GP** is the combination of [cgan](https://arxiv.org/abs/1411.1784) and [wgan-gp](https://arxiv.org/pdf/1704.00028). It can be used to generate samples of a particular class.

### Datasets ###

Mnist dataset is used in the experiment.

### Usage ###
python3 cwgan_gp.py

### Result ###
![avatar](/images/sample_image.png)

### Generate image of a specific class ###
Use the code *wgan.generate\_images(class\_names)*.

### Defining a new task for cWGAN-GP ###
Need to change *img\_rows*, *img\_cols*, *channels*, *nclasses*, load your own dataset and change the network structure to corresponding size.