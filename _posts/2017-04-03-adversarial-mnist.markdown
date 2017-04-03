---
layout: post
title:  "Exploring and Thwarting Adversarial Examples in MNIST"
excerpt: "In this post I explore adversarial examples in the popular
handwritten digit recognition dataset MNIST, and also present a simple
technique for thwarting such adversarial examples."
date:   2017-04-03 00:08:00 +0900
mathjax: true
---

<style type="text/css">
  .post-header h1 {
    font-size: 35px;
  }

  table {
    border-spacing: 0;
  }

  tr:nth-child(even) {
    background-color: #eee;
  }

  th {
    padding: 0 10px;
  }

  table.results td:nth-child(2),
  table.results td:nth-child(3) {
    text-align: center;
  }

  td {
    padding: 0 5px;
  }

  tr.highlight {
    background-color: yellow;
  }

  tr.inner_header {
    height: 50px;
    font-weight: 500;
  }

  #fc_0_02_results table, #fc_fcn_0_02_results table,
  #results_0_20 table, #results_0_05 table {
    width: 615px;
  }

  #fc_0_02_results .tablecap, #fc_fcn_0_02_results .tablecap,
  #results_0_20 .tablecap, #results_0_05 .tablecap {
    width: 615px;
  }

  #results_0_20 table, #results_0_05 table {
    width: 100%;
  }

  #results_0_20 .tablecap, #results_0_05 .tablecap {
    width: 100%;
  }

  .mono {
    font-family: monospace;
  }

</style>

<div class="imgcap">
  <img src="/assets/mnist_adv/examples_with_noise_0.05.png">
  <div class="thecap">
    Are you sure those numbers are what you think they are? Images on the left of each set are original images, middle images are amplified depictions of the noise, and images on the right are crafted adversarial images (original+noise). Classification probabilities are showed for each image, on a simple conv net which gets greater than 99.25% accuracy on MNIST. The maximum noise limit here is 0.05.</div>
</div>

### Abstract

The grayscale MNIST handwritten digits dataset, while much smaller and simpler than ImageNet, is still very susceptible to adversarial examples, which are specifically crafted images meant to trick neural nets into misclassifications. Adding multivariate Gaussian noise layers into the neural net at various layers is one way of making the net more resilient to adversarial examples, often with negligible impact on model performance.

### Intro

The idea of "tricking" convolutional neural nets is not new. These neural nets, which have shown amazing results for many different computer vision applications, are very susceptible to making certain types of mistakes. Goodfellow et. al were some of the first to initially describe the problem in detail in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). And Andrej Karpathy did a fantastic job of
describing the issue [here](https://karpathy.github.io/2015/03/30/breaking-convnets/).
(Also, I borrowed much of his blog structure and CSS, so Andrej, big thanks!!). Essentially, imperceptibly small changes on multiple pixels can add up to one large mistake by the neural net, with the result being that it misclassifies a bus as an ostrich, or the number "3" as the number "6".

Rather than focusing on high-dimensional dot products, the way I like to think of this problem is simpler - neural net image classifiers operate in an extremely high-dimensional space, but the divisions between classes are too "sharp" in that high-dimensional space, so it's easy to push the classifier away from the actual class and towards a wrong "adversarial" class. On the other hand, humans have a "fuzzier" classification model that is obviously much more resilient to these tiny changes, because our visual system "averages" them out. Below is a simple visualization for how to think about the "sharpness" of a classifier, with the image on the left showing a much "sharper" classifier than the image on the right.

<div class="imgcap">
  <img src="/assets/mnist_adv/classification_visualization.png">
  <div class="thecap">
    3D visualization of high dimensional classification "sharpness" for two ImageNet classes, "panda" and "gibbon". On the left, the classifier has a very sharp classification range that hugs the line, which represents the "true" class projection. On the right, the classifier has a fuzzier classification range.</div>
</div>

Goodfellow et. al use the adversarial examples themselves in training as a way to get better ultimate model performance, but I wanted to ask a different question: **Can we change the structure of the neural net itself in order to thwart adversarial examples?**

(Skip the math, code, and numerical results, and [take me straight to the discussion](#discussion).)

### Methodology

#### Model structure

Since the problem is that the high-dimensional classifications are too "sharp", I decided it would be interesting to see what happens if we make the activations "fuzzier". And the first idea that popped into my head was to take the whole concept of adding noise to inputs for adversarial example generation and instead just add random noise to the inputs before feeding them into the neural net for training (this is different from Goodfellow et. al, who added specifically crafted "adversarial" noise as part of the training step).

Below are some standard convolutional neural nets for classifying MNIST images.
On the left is a convolutional neural net with two convolutional layers
followed by a fully connected layer, and on the right is a fully
convolutional model (i.e., no fully connected layers).

<div class="imgcap">
  <img src="/assets/mnist_adv/mnist_models.png"
   height="600px">
  <div class="thecap">
    A convolutional net ending with a fully-connected layer (left), and a fully convolutional neural net (right) for classifying MNIST images.
  </div>
</div>

The idea, then, is to modify the above simple models by adding an input distortion layer. Since the input for MNIST is a 28x28 pixel image (or equivalently, a length 784 vector if we flatten it out), the simplest way to distort the input is by adding multivariate Gaussian noise to each pixel (note that the noise is multivariate, because we want to distort each dimension independently rather than move all pixel values by the same random amount). Frameworks like Tensorflow, Torch, and Theano are actually general computational graph frameworks and thus allow for arbitrary layers in neural nets other than the standard convolutional, max pooling, and fully-connected layers that we're familiar with. And since once you design your own layer in a modular way, you can stick it anywhere in the net, I decided to experiment with adding multivariate Gaussian noise not only after the input, but also after the convolutional layers and the final fully-connected layer. The idea is to "smooth" out the activations like in the earlier visualization. Below is the model structure:

<div class="imgcap">
  <img src="/assets/mnist_adv/distortion_layers.png"
   height="800px">
  <div class="thecap">
    A conv net showing Gaussian distortion layers after input,
    conv1, conv2, and the final fully connected layer.
  </div>
</div>

As you can see, there are distortion layers after the input, the convolutional layers (after the max pooling), and the fully connected layer. Each layer has conditionals controlling whether or not distortion is applied to the activations (for the first distortion layer, the "activations" are simply the original input images). And just like standard dropout layers, we only apply distortion during training but not during testing.

Given this model structure, we can now efficiently experiment with many different permutations of distortion layers mixed into the standard convolutional neural net structures that we're familiar with.

#### Efficient adversarial example generation

As for generating adversarial examples, we use the "fast gradient sign method" described in [Goodfellow et. al](https://arxiv.org/abs/1412.6572). The idea is as follows: our cost function (i.e., the "loss") tells us how wrong our predictions are, and the gradient of the cost function tells us what perturbations we can make in order to decrease the cost. For normal neural net training, we use the gradient of the loss in order to update the model parameters (i.e., weights and biases). But we can similarly hold the parameters constant and instead perturb the *inputs* according to the gradient of the loss, with the same objective of minimizing the loss. Since a prediction from a neural net requires inputs and labels, in order to generate adversarial examples, we just choose a different label from the actual label and send that in with the inputs to a pre-trained neural net. So if MNIST says the 1024th training image is a "6", we can instead choose another label (let's say "3"), get the loss for that class with respect to that input, and then perturb the image in the direction opposite the gradient, which mathematically is the direction to go in order to decrease the loss.

At this point it should be clear that generating adversarial examples is entirely analogous to training a neural net, with the only difference being that during training we update the model parameters, and during adversarial example generation, we update the inputs.

Explicitly, as discussed in [Stanford CS231N's Linear Classification primer](http://cs231n.github.io/linear-classify/#softmax), the loss for an individual example is $$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right)$$,
where we are using the notation \\(f_j\\) to mean the j-th element of the vector of class scores \\(f\\). The full loss over the dataset is the mean of \\(L_i\\) over all training examples together with a regularization term \\(R(W)\\) (which we're ignoring below for simplicity).
The function \\(g_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}} \\) is the **softmax function**, using \\(g\\) here in order to disambiguate from the class scores \\(f\\).
The softmax function takes a vector of arbitrary real-valued scores (in \\(z\\)) and squashes it to a vector of values between zero and one that sum to one. That means we can treat the outputs of the softmax function like "probabilities", because they sum to one. In the case of a deep neural net, that "vector of arbitrary real-valued scores in \\(z\\)" refers to the logits that are spit out at the final step (before softmax) of the neural net.

So for example, for a three-class classification problem, let's say we have:

```python
logits = [2, 10, 8]
```

Then the softmax output looks like:

```python
class_probs = np.exp(logits)/np.sum(np.exp(logits))
print('Class probabilities:', [float('%.4f' % score) for score in class_scores])
print('Sum = %f' % np.sum(class_probs))
```
Has output:

```
Class probabilities: [0.0003, 0.8805, 0.1192]
Sum = 1.0
```

Remember that the loss for a single example is $$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right)$$, or $$-\log({softmax({f_{y_i}})})$$. The reason the loss is the negative log is apparent if we look at a graph:

<div class="imgcap">
  <img src="/assets/mnist_adv/lnx.png" height="300px">
  <div class="thecap">
  </div>
</div>

The "loss" measures how bad our predictions are. So if the class probability for the target class is close to 1, then the cost/loss should be low (since we're predicting correctly), and if the class probability is close to 0, then the loss should be high, because we want to penalize that bad prediction.

Okay. So, the gradient of the loss with respect to the input \\(x_i\\) is

$$
\nabla_{x_i} L_i = \nabla_{x_i}
  \left(
    -\log
      \left(
        \frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }
      \right)
  \right)
$$

In Tensorflow, we typically have a tensor for the loss that looks like:

```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=logits))
```
We can use the low-level automatic differentiation that Tensorflow provides to get the symbolic gradient of this loss with respect to the input by:

```python
adversarial_grad = tf.gradients(loss, inputs)
```
But ... we have a little bit of an issue. The loss is averaged over all the inputs, thus the above code for the loss tensor really means

$$
loss_{batch} = -\frac{1}{D} \sum_{k=1}^D \log
\left(
  \frac{e^{f_{y_{k}}}}{ \sum_j e^{f_{k_j}}}
\right)
$$

where \\(D\\) is the number of samples in the current mini-batch, and now \\(f_{k_j}\\) refers to the j-th element of the vector of class scores for the k-th sample in the mini-batch, and \\(y_k\\) is the true class for the k-th sample, meaning $$\frac{e^{f_{y_{k}}}}{ \sum_j e^{f_{k_j}}}$$ is the class score for the correct class of the k-th sample in the mini-batch (whew!).  Expanding, this is equivalent to

$$
loss_{batch} = -\frac{1}{D} \left(
\log\left(\frac{e^{f_{y_1}}}{ \sum_j e^{f_{1_j}}}\right)+
\log\left(\frac{e^{f_{y_2}}}{ \sum_j e^{f_{2_j}}}\right)+
\ldots+
\log\left(\frac{e^{f_{y_d}}}{ \sum_j e^{f_{d_j}}}\right)
\right)
$$

This means our <span class="mono">adversarial_grad</span> above gives us the gradient with respect to the mini-batch, with influence from every sample in the batch, whereas what we really want to do is figure out the best perturbation to make *for each individual example*. Goodfellow wrote a paper about this problem in [Efficient Per-Example Gradient Computations](https://arxiv.org/abs/1510.01799), explaining that the simplest brute force approach to solving this problem is to just compute the gradient D times, each for a batch size of 1, but for our case, since we control the model directly, there's actually a super simple solution. All we have to do is split up the loss into two steps:

```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=logits)
loss = tf.reduce_mean(cross_entropy)
```

and then:

```python
adversarial_grad = tf.gradients(cross_entropy, inputs)
```
Now <span class="mono">adversarial_grad</span> will be of the shape <span class="mono">[batch_size, input_dimensions]</span> or <span class="mono">[D, 784]</span> in our case, and contains

$$
\nabla_{x_i} L_i = \nabla_{x_i}
  \left(
    -\log
      \left(
        \frac{e^{f_{y_i}}}
             { \sum_j e^{f_j} }
      \right)
  \right)
$$

for each \\(x_i\\) in the mini-batch \\(i=1..D\\)! This is exactly what we want, since in the brute force case of a batch size of 1, we have:

$$
loss_{batch} = -\frac{1}{D} \sum_{k=1}^D \log
\left(
  \frac{e^{f_{y_{k}}}}{ \sum_j e^{f_{k_j}}}
\right) =

-\log\left(\frac{e^{f_{y_1}}}{ \sum_j e^{f_{1_j}}}\right)
$$

and from above, the gradient of the loss with respect to one input sample \\(x_i\\) is

$$\nabla_{x_i} L_i = \nabla_{x_i} \left(-\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right)\right)$$

which is exactly what we have in our gradient vector of shape <span class="mono">[D, 784]</span>, with each row i containing the gradient of the loss for sample i with respect to only sample i.

We can thus roll our own gradient descent using this gradient tensor as follows:

```python
# inputs contains [D, 784] MNIST images
# adversarial_targets is one-hot representation of shape [D, 10] for the
#     adversarial classes we are trying to create.

noise = np.zeros_like(inputs)
for i in range(max_iters):
    # Clip noisy inputs to make sure they are valid
    noisy_inputs = np.clip(inputs+noise, 0.0, 1.0)

    # Get the gradient of the loss per example for the current inputs
    gradients = sess.run(adversarial_grad, feed_dict={inputs: noisy_inputs,
                                                      targets: adversarial_targets})
    # Update the noise by performing gradient descent
    noise -= learning_rate*gradients

    # Make sure the noise doesn't exceed predetermined limit
    noise = np.clip(noise, -noise_limit, noise_limit)
```

This efficiently computes adversarial examples a batch at a time, optimizing each example independently! Note that we add the negative of the gradients to the current noise because the gradient of a function shows the direction to move in order to increase the value of the function, but we're trying to *minimize* the loss (which *maximizes* our class scores), so we have to move in the opposite direction. Some quick experimentation found $$1\mathrm{e}{-1}$$ to be a good learning rate for MNIST.

### Results

Alright, now that we have our model structure and an efficient way of computing adversarial examples, let's look at some results.

Model layers as seen below are as follows:

- **input**: Unmodified input layer, i.e., batches of MNIST 28x28 images.
- **conv1**: First convolutional layer, followed by a max pool layer.
- **conv2**: Second convolutional layer, followed by a max pool layer.
- **fc**: Fully connected layer.
- **d0.xx**: Multivariate Gaussian noise layer with standard deviation of
0.xx on each dimension of the input (e.g., d0.50 means noise with a
standard deviation of 0.50 on each dimension).
- **softmax** Linear readout layer followed by softmax.

First, I was curious if the net would be able to learn on noisy inputs, but still perform well on clean test inputs. Below are examples of distorted MNIST images with different levels of noise. The noise level corresponds to the standard deviation of the normal distribution for all dimensions of the input (i.e., each dimension is using the same size normal distribution, but each dimension samples from that same sized normal distribution independently of all other dimensions).

<div class="imgcap">
  <img src="/assets/mnist_adv/input_distortion.png">
  <div class="thecap">
    Different levels of multivariate Gaussian noise applied to MNIST inputs.
  </div>
</div>

Lower levels of noise are still easy for us to read, but once you get above 0.5 standard deviation, the numbers become increasingly difficult to make out. Can the model still learn? The answer is a resounding yes! Even at super high levels of distortion in the input images, somehow the neural net still learns reasonably well how to classify clean images. Here is the test set performance for five different levels of input distortion:

<div markdown="1" id="fc_0_02_results">

|----------------------------------------------------------------------|:---------------:|
|                                   Model                               | Test set accuracy |
|----------------------------------------------------------------------|:---------------:|
|                          input - d0.10 - conv1 - conv2 - fc - softmax |   0.992100 |
|                          input - d0.30 - conv1 - conv2 - fc - softmax |   0.992400 |
|                          input - d0.50 - conv1 - conv2 - fc - softmax |   0.992100 |
|                          input - d0.70 - conv1 - conv2 - fc - softmax |   0.985200 |
|                          input - d0.90 - conv1 - conv2 - fc - softmax |   0.975900 |
|----------------------------------------------------------------------|:---------------:|

  <div class="tablecap">
	    Test set performance for models with different levels of input distortion during training. Noise limit for adversarial examples is 0.02.
  </div>
</div>

Note that I didn't try very hard to get better performance out of these models. Each one is trained for a maximum of 50 epochs using the Adam algorithm for gradient descent back propagation (<span class="mono">AdamOptimizer</span> in Tensorflow) with an initial learning rate of $$1\mathrm{e}{-4}$$.

Look again at the distorted inputs above with noise levels of 0.90. These are completely unreadable to the human eye, but the neural net still learns with 97.5% accuracy how to read numbers from these noisy inputs! Fascinating.

**So what effect does distorting the inputs have on the ease of generating adversarial images?** Average adversarial target score (shown as "Avg target score" in the table) is computed as follows: we generate 100 different adversarial examples for each model, each with a max noise limit of 0.2. The average target score is just the average over all generated adversarial examples after performing the above simple gradient descent for 1000 iterations. Target classes are chosen randomly from the non-labels (i.e., if the label is a "3", then we randomly choose an adversarial target class that's a number from 0 to 9 that's not 3). Models ending with a fully-connected layer as well as fully convolutional models are shown below.

<div id="fc_fcn_0_02_results">
<table class="results">
  <thead>
    <tr>
      <th>Model</th>
      <th>Test set accuracy</th>
      <th>Avg target score</th>
    </tr>
  </thead>
  <tbody>
    <tr class="inner_header"><td colspan="3">Fully connected models:</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.971253</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.703507</td></tr>
    <tr class="highlight"><td>input - d0.50 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.356592</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - softmax</td><td>0.985200</td><td>0.226507</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - softmax</td><td>0.975900</td><td>0.156984</td></tr>
    <tr class="inner_header"><td colspan="3">Fully convolutional models:</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.975981</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.813792</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - softmax</td><td>0.986100</td><td>0.549809</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - softmax</td><td>0.977000</td><td>0.379621</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - softmax</td><td>0.965400</td><td>0.394852</td></tr>
  </tbody>
</table>
  <div class="tablecap">
Test set performance and average adversarial target score for models with input distortion, both fully convolutional and ending with a fully-connected layer. Noise limit for adversarial examples is 0.02.
  </div>
</div>


Surprisingly, this very simple change to training has a profound effect on adversarial images - with very little performance hit, the average final score for our adversarial images drops precipitously. Input distortion with a standard deviation of 0.50 in the model ending with a fully-connected layer particularly stands out, since the test set accuracy doesn't decrease at all, yet the average adversarial image target score has dropped from 97% to 35%. Fully convolutional models with input distortion have a bit worse performance, but still do a solid job of thwarting adversarial examples compared to the plain models, and are also substantially smaller sized models than the ones ending with fully-connected layers.

#### Going overboard with model permutations

Once I automated the training and adversarial example generation, I went a bit overboard and made 97 different models with different combinations of distortion layers and different levels of distortion. The table below shows the test set accuracy and average adversarial target scores for some of the better models, with a noise limit of 0.2. I chose 0.2 noise limit for this table because there's a very nice performance spread. (At higher noise limits like 0.3-0.5, we can easily generate good adversarial examples for all models.) Note that higher test accuracy is good, and lower average adversarial target scores are also good. Full results for all 97 models can be seen in [the Appendix](#full_results_0_20).

<div id="results_0_20">
<table class="results">
  <thead>
    <tr>
      <th>Model</th>
      <th>Test set accuracy</th>
      <th>Avg target score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.356592</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.989500</td><td>0.435470</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - d0.30 - softmax</td><td>0.986100</td><td>0.358353</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - softmax</td><td>0.985200</td><td>0.226507</td></tr>
  </tbody>
</table>
  <div class="tablecap">
Some standout models from adversarial noise limit of 0.2.
  </div>
</div>

Here are some results from a different run with an adversarial noise limit of 0.05. The full table can be found in [the Appendix](#full_results_0_05), but there were a few standouts:

<div id="results_0_05">
<table class="results">
  <thead>
    <tr>
      <th>Model</th>
      <th>Test set accuracy</th>
      <th>Avg target score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991200</td><td>0.000005</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.990300</td><td>0.000588</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.972900</td><td>0.000000</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.978800</td><td>0.000001</td></tr>
  </tbody>
</table>
  <div class="tablecap">
Some standout models from adversarial noise limit of 0.05.
  </div>
</div>

Some of these models are a bit more exotic than just a simple input distortion. The first model in the 0.05 noise limit table has distortion layers after the input and the fully-connected layer. The next has very light distortion layers after the input and both convolutional layers. The third is the same structure as the second, but with slightly higher distortion levels. And the fourth has a stronger distortion layer after the input and the first convolutional layer. As you can see, the first two have quite good test set performance while being extremely resilient to adversarial examples at this noise limit (0.05). The third and fourth are even more resilient to adversarial examples, but take a bit of a performance hit.

<a id="discussion"></a>
### Discussion

So what does it all mean? Is this simple technique really all that's needed to combat adversarial examples? And is that even a worthwhile goal?

Well, first it's important to understand that while we can provide resilience against adversarial examples *on average*, that doesn't mean that the model is completely resilient to adversarial examples. If our goal is to just find one really good adversarial example, then we can usually do so (especially with our efficient gradient computations described above). For example, here are two that I found at a 0.05 noise limit for the *input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - softmax* model from above, which had an average target score of zero in the initial run:

<div class="imgcap">
  <img src="/assets/mnist_adv/exotic_examples.png">
  <div class="thecap">
    A very strong adversarial example and a reasonably strong adversarial example found for a seemingly impervious model.
  </div>
</div>

The first example is a very strong adversarial example - the model incorrectly thinks it's a 0 with nearly 100% accuracy. The second is reasonably strong at about 75%. But actually, those were the only adversarial example found in the entire run, trying with 1000 different images to generate adversarial examples. The next best example has a target class score of 10% (which means it's still predicting the correct class with a score of 90%), and the target class scores quickly trail off to zero. So this particular model structure is incredibly resilient to adversarial examples *at this noise limit*. This is pretty cool! (BTW, if the adversarial images above look exactly like the original images, zoom in until you can see the noise. I promise it's there.)

It's also interesting to compare the adversarial noise generated for this model compared with the examples at the top of the post, which were generated against a simple *input - conv1 - conv2 - fc - softmax* model. The model with the three distortion layers appears to force more "splotchy" adversarial noise. It's difficult to speculate why this is, but it's a curious result.

The implication of all this is that for this specific model, an "attacker" would need to craft an adversarial example at a higher noise limit, and the higher noise limits are definitely going to be more noticeable, so they're less likely to "trick" a human viewer. From a security perspective, this is precisely our goal! However, we've also made a tradeoff here - some of these resilient models such take a performance hit. Is it worth giving up a few percentage points of accuracy (when even tenths of percent improvement merit widespread accolades these days) just to resist adversarial examples?

Let's take a look at another model from above: *input - d0.30 - conv1 - conv2 - fc - d0.30 - softmax*, which had pretty stellar performance of 0.991200 while only allowing a barely positive average target score of 0.000005. Running the same test as above on 1000 random adversarial examples, we only successfully find four:

<a id="exotic_examples_input_0_30_fc_0_30" />
<div class="imgcap">
  <img src="/assets/mnist_adv/exotic_examples_input_0.30_fc_0.30.png">
  <div class="thecap">
    Adversarial examples found for a conv net with distortion after the input layer and final fully-connected layer.
  </div>
</div>

Thus, this model is extremely resilient to adversarial examples while also maintaining nearly the same model performance as the simple conv net without distortion layers!

#### Neuroscientific musings

Adversarial examples also provide us with interesting insights into how visual processing in the human brain differs from that of convolutional neural nets. It should be clear by now that neural nets *do not "see" at all like humans*. I'm going to say that again, because it bears repeating: neural nets that classify or detect objects in images *do so in a way that is fundamentally different from humans*. The definition of adversarial examples used in the AI community usually refers to examples that humans can't distinguish, but that cause incorrect predictions by the neural net. So while humans see the number "3", or a panda, the neural net sees an "8", or a gibbon.

One of the reasons for this difference is that the human visual system is far more complex than a convolutional neural net, which at its core is just a series of matrix multiplications. The human visual system is "holistic", which means it averages out tiny imperfections and differences and makes a weighted decision based more on higher level structures (e.g., for an "8", it's looking for something approximating two loops, one on top of the other). On the other hand, the convolutional neural nets are the exact opposite - while we arrange them in layers for the express purpose of simulating how we think the human visual system might work, as described in the intro, these types of neural nets are *extremely* vulnerable to lots of small imperfections.

Another important difference between the human visual system and convolutional neural nets has to do with perceptual constancy. When considering the *difference* between colors, neural nets don't see any difference between the difference of two pixels with values 0.10 and 0.20 or 0.70 and 0.80 - both are a numerical difference of 0.10. But humans absolutely see a difference. Look again at the adversarial examples [above](#exotic_examples_input_0_30_fc_0_30) for the *input - d0.30 - conv1 - conv2 - fc - d0.30 - softmax* model. The noise is hard to perceive in the adversarial examples unless you really look closely (although it also depends on the display characteristics of your device). But now take a look at what happens if we invert the images:

<a id="exotic_examples_input_0_30_fc_0_30" />
<div class="imgcap">
  <img src="/assets/mnist_adv/exotic_examples_input_0.30_fc_0.30_inverted.png">
  <div class="thecap">
    Adversarial examples found for a conv net with distortion after the input layer and final fully-connected layer.
  </div>
</div>

The noise in the adversarial examples now looks much more noticeable, and most humans would not be fooled into thinking the images are unmodified. The reason has to do with something called *lightness constancy*, whereby the human visual system averages over total illumination in a scene when perceiving different levels of brightness. It's the source for a number of great visual illusions, one of which is shown below, and more which [can be seen here](http://www.cns.nyu.edu/~david/courses/perception/lecturenotes/brightness-contrast/brightness-contrast.html).

<div class="imgcap">
  <img src="/assets/mnist_adv/brightness_constancy.jpg"
   height="400px">
  <div class="thecap">
    Example of optical illusion caused by lightness constancy in human visual system. But conv net don't care. Credit to: <a href="http://www.cns.nyu.edu/~david/courses/perception/lecturenotes/brightness-contrast/brightness-contrast.html">Professor David Heeger from NYU</a>.
  </div>
</div>

Currently, neural nets used for computer vision don't even attempt to model neurovisual processes such as lightness and color constancy. Rather, we feed them inputs and labels and let the training process decide via simple optimization what the model parameters should be. Can we improve performance by building nets that more closely resemble the idiosyncrasies of our own visual systems? Should the fragility of convolutional neural nets to small input perturbations worry us? Will larger and more complicated models be more or less susceptible to such adversarial attacks?

Studying how these highly performant neural nets perform and fail provides us with valuable insight into the differences between the human visual system and the mathematical model of it that we are striving to create.

### Future work

This paper looked at the effect on adversarial example generation of adding multivariate Gaussian noise at different levels of a convolutional neural net. This sort of distortion of the inputs and the different level activations can have profound effects on the resilience of a convolutional neural net to adversarial examples.

Some areas for future study are:

- **Push the model performance for specific resilient models.** In some of the resilient models, the test performance is very close to the test performance of the plain models. Can we optimize the hyperparameters to get equal or better performance?
- **Different levels of distortion at different places in the conv net.** In this paper, if a model had multiple distortion layers, all layers used the same distortion. What if we varied it depending on the layer?
- **Different levels of distortion for each dimension.** In this paper, distortion layers apply Gaussian noise at the same level for each dimension of the input (e.g., for the input layer, each of the 784 different dimensions of the input are distorted by a normal distribution of the same size, even though the dimensions all get perturbed independently). But we have the flexibility to choose different distributions for each dimension.
- **Base the distortion levels on the actual distributions of the activations.** In this paper, the distortion levels were chosen beforehand and are not related to the actual distributions of the activations. But we can very easily train a conv net, look at the actual distributions of the activations at the layer we want to apply distortion to, and craft a distortion layer based on the activations.
- **Change distortion levels during training.** In this paper, the distortion level is fixed at the start of training. But we can modify the amount of distortion at each step in training if we choose. Would something like exponential decay of the distortion magnitude yield better results?
- **Different data sets.** Obviously, it would be interesting to see how well this approach works at combating adversarial examples on larger datasets such as ImageNet, where inputs are larger sized and have all three color channels.

### Additional reading

- [Breaking Linear Classifiers on ImageNet](https://karpathy.github.io/2015/03/30/breaking-convnets/)
- [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)
- [Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572)
- [Assessing Threat of Adversarial Examples on Deep Neural Networks](https://arxiv.org/abs/1610.04256)
- [Using Non-invertible Data Transformations to Build Adversarial-Robust Neural Networks](https://arxiv.org/abs/1610.01934)

### Appendix

<a id="full_results_0_20"></a>

Results for all models, with an adversarial noise limit of 0.02.

#### By test set accuracy:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Test set accuracy</th>
      <th>Avg target score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.815635</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.899505</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.703507</td></tr>
    <tr><td>input - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.990381</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.992300</td><td>0.927010</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.992300</td><td>0.898346</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.356592</td></tr>
    <tr class="highlight"><td>input - d0.10 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.971253</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.992000</td><td>0.956228</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.991900</td><td>0.709173</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - d0.10 - softmax</td><td>0.991900</td><td>0.684761</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.991700</td><td>0.958510</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991700</td><td>0.947238</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.991400</td><td>0.924330</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - softmax</td><td>0.991400</td><td>0.887286</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.991200</td><td>0.829334</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991200</td><td>0.602859</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.991100</td><td>0.958066</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.990700</td><td>0.859379</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.990300</td><td>0.659377</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - softmax</td><td>0.989700</td><td>0.978254</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.989500</td><td>0.435470</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.989400</td><td>0.678443</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.989400</td><td>0.739782</td></tr>
    <tr><td>input - conv1 - conv2 - softmax</td><td>0.988900</td><td>0.954959</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.988400</td><td>0.697563</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.813792</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.975981</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.986900</td><td>0.452798</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - softmax</td><td>0.986600</td><td>0.894109</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - softmax</td><td>0.986200</td><td>0.787606</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - d0.30 - softmax</td><td>0.986100</td><td>0.358353</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - softmax</td><td>0.986100</td><td>0.549809</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.985800</td><td>0.744208</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.985800</td><td>0.399978</td></tr>
    <tr class="highlight"><td>input - d0.70 - conv1 - conv2 - fc - softmax</td><td>0.985200</td><td>0.226507</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.985100</td><td>0.312749</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - softmax</td><td>0.984700</td><td>0.664513</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - softmax</td><td>0.981000</td><td>0.837506</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.980100</td><td>0.612057</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.979800</td><td>0.679863</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.979000</td><td>0.825392</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.978800</td><td>0.215110</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.978800</td><td>0.709367</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - softmax</td><td>0.977700</td><td>0.797510</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - softmax</td><td>0.977000</td><td>0.379621</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - softmax</td><td>0.976600</td><td>0.867573</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - softmax</td><td>0.976300</td><td>0.645693</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - softmax</td><td>0.975900</td><td>0.156984</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.973200</td><td>0.274042</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.973100</td><td>0.337120</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.972900</td><td>0.326195</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.972900</td><td>0.798006</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.972300</td><td>0.629216</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - softmax</td><td>0.971500</td><td>0.318552</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - d0.50 - softmax</td><td>0.967700</td><td>0.179698</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - softmax</td><td>0.965400</td><td>0.394852</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.965300</td><td>0.258568</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - softmax</td><td>0.964000</td><td>0.718437</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - softmax</td><td>0.960600</td><td>0.719473</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.960400</td><td>0.441536</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.958800</td><td>0.213422</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.958100</td><td>0.299930</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - softmax</td><td>0.952900</td><td>0.661862</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.950900</td><td>0.497812</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.938000</td><td>0.609531</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.936800</td><td>0.284702</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - softmax</td><td>0.936000</td><td>0.334070</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.935200</td><td>0.239760</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.933000</td><td>0.528727</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - d0.70 - softmax</td><td>0.929400</td><td>0.190118</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.923500</td><td>0.380114</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - softmax</td><td>0.923200</td><td>0.406704</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.921400</td><td>0.249861</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - softmax</td><td>0.920900</td><td>0.481427</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.920300</td><td>0.282721</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - softmax</td><td>0.898600</td><td>0.582429</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.898200</td><td>0.706772</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - softmax</td><td>0.887600</td><td>0.519188</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.886300</td><td>0.669690</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.863900</td><td>0.648991</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.857100</td><td>0.308378</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.850100</td><td>0.328414</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - softmax</td><td>0.829700</td><td>0.556545</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.827300</td><td>0.558393</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.825500</td><td>0.440884</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - softmax</td><td>0.824500</td><td>0.415634</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.776600</td><td>0.488673</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.776200</td><td>0.481098</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.749500</td><td>0.499989</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.690500</td><td>0.418346</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.171600</td><td>0.139717</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.118800</td><td>0.156104</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - d0.90 - softmax</td><td>0.113500</td><td>0.136867</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.112400</td><td>0.106029</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.098200</td><td>0.151960</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.083900</td><td>0.164867</td></tr>
  </tbody>
</table>

#### By avg target score:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Test set accuracy</th>
      <th>Avg target score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.112400</td><td>0.106029</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - d0.90 - softmax</td><td>0.113500</td><td>0.136867</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.171600</td><td>0.139717</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.098200</td><td>0.151960</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.118800</td><td>0.156104</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - softmax</td><td>0.975900</td><td>0.156984</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.083900</td><td>0.164867</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - d0.50 - softmax</td><td>0.967700</td><td>0.179698</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - d0.70 - softmax</td><td>0.929400</td><td>0.190118</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.958800</td><td>0.213422</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.978800</td><td>0.215110</td></tr>
    <tr class="highlight"><td>input - d0.70 - conv1 - conv2 - fc - softmax</td><td>0.985200</td><td>0.226507</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.935200</td><td>0.239760</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.921400</td><td>0.249861</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.965300</td><td>0.258568</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.973200</td><td>0.274042</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.920300</td><td>0.282721</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.936800</td><td>0.284702</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.958100</td><td>0.299930</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.857100</td><td>0.308378</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.985100</td><td>0.312749</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - softmax</td><td>0.971500</td><td>0.318552</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.972900</td><td>0.326195</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.850100</td><td>0.328414</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - softmax</td><td>0.936000</td><td>0.334070</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.973100</td><td>0.337120</td></tr>
    <tr class="highlight"><td>input - d0.50 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.356592</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - d0.30 - softmax</td><td>0.986100</td><td>0.358353</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - softmax</td><td>0.977000</td><td>0.379621</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.923500</td><td>0.380114</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - softmax</td><td>0.965400</td><td>0.394852</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.985800</td><td>0.399978</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - softmax</td><td>0.923200</td><td>0.406704</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - softmax</td><td>0.824500</td><td>0.415634</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.690500</td><td>0.418346</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.989500</td><td>0.435470</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.825500</td><td>0.440884</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.960400</td><td>0.441536</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.986900</td><td>0.452798</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.776200</td><td>0.481098</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - softmax</td><td>0.920900</td><td>0.481427</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.776600</td><td>0.488673</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.950900</td><td>0.497812</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.749500</td><td>0.499989</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - softmax</td><td>0.887600</td><td>0.519188</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.933000</td><td>0.528727</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - softmax</td><td>0.986100</td><td>0.549809</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - softmax</td><td>0.829700</td><td>0.556545</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.827300</td><td>0.558393</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - softmax</td><td>0.898600</td><td>0.582429</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991200</td><td>0.602859</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.938000</td><td>0.609531</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.980100</td><td>0.612057</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.972300</td><td>0.629216</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - softmax</td><td>0.976300</td><td>0.645693</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.863900</td><td>0.648991</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.990300</td><td>0.659377</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - softmax</td><td>0.952900</td><td>0.661862</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - softmax</td><td>0.984700</td><td>0.664513</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.886300</td><td>0.669690</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.989400</td><td>0.678443</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.979800</td><td>0.679863</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - d0.10 - softmax</td><td>0.991900</td><td>0.684761</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.988400</td><td>0.697563</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.703507</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.898200</td><td>0.706772</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.991900</td><td>0.709173</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.978800</td><td>0.709367</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - softmax</td><td>0.964000</td><td>0.718437</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - softmax</td><td>0.960600</td><td>0.719473</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.989400</td><td>0.739782</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.985800</td><td>0.744208</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - softmax</td><td>0.986200</td><td>0.787606</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - softmax</td><td>0.977700</td><td>0.797510</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.972900</td><td>0.798006</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.813792</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.815635</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.979000</td><td>0.825392</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.991200</td><td>0.829334</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - softmax</td><td>0.981000</td><td>0.837506</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.990700</td><td>0.859379</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - softmax</td><td>0.976600</td><td>0.867573</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - softmax</td><td>0.991400</td><td>0.887286</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - softmax</td><td>0.986600</td><td>0.894109</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.992300</td><td>0.898346</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.899505</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.991400</td><td>0.924330</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.992300</td><td>0.927010</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991700</td><td>0.947238</td></tr>
    <tr><td>input - conv1 - conv2 - softmax</td><td>0.988900</td><td>0.954959</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.992000</td><td>0.956228</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.991100</td><td>0.958066</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.991700</td><td>0.958510</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.971253</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.975981</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - softmax</td><td>0.989700</td><td>0.978254</td></tr>
    <tr><td>input - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.990381</td></tr>
  </tbody>
</table>

<a id="full_results_0_05"></a>
Here's another table showing results of all models, but this time with a noise limit of 0.05, which is the same limit as the figure at the top of the post. 0.05 is barely perceptible to us.

#### By test set accuracy:

<table class="results">
  <thead>
    <tr>
      <th>Model</th>
      <th>Test set accuracy</th>
      <th>Avg target score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.012553</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.005734</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.027068</td></tr>
    <tr><td>input - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.013994</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.992300</td><td>0.029956</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.992300</td><td>0.009995</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.000589</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.033148</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.992000</td><td>0.015429</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.991900</td><td>0.013715</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - d0.10 - softmax</td><td>0.991900</td><td>0.019433</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.991700</td><td>0.022410</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991700</td><td>0.016735</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.991400</td><td>0.008776</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - softmax</td><td>0.991400</td><td>0.010776</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.991200</td><td>0.000821</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991200</td><td>0.000005</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.991100</td><td>0.002841</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.990700</td><td>0.010305</td></tr>
    <tr class="highlight"><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.990300</td><td>0.000588</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - softmax</td><td>0.989700</td><td>0.018690</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.989500</td><td>0.000731</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.989400</td><td>0.000027</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.989400</td><td>0.009515</td></tr>
    <tr><td>input - conv1 - conv2 - softmax</td><td>0.988900</td><td>0.063591</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.988400</td><td>0.022035</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.013762</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.049586</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.986900</td><td>0.010000</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - softmax</td><td>0.986600</td><td>0.037620</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - softmax</td><td>0.986200</td><td>0.001474</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - d0.30 - softmax</td><td>0.986100</td><td>0.002538</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - softmax</td><td>0.986100</td><td>0.014919</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.985800</td><td>0.001278</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.985800</td><td>0.010068</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - softmax</td><td>0.985200</td><td>0.019258</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.985100</td><td>0.010018</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - softmax</td><td>0.984700</td><td>0.009997</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - softmax</td><td>0.981000</td><td>0.006804</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.980100</td><td>0.010240</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.979800</td><td>0.030138</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.979000</td><td>0.019071</td></tr>
    <tr class="highlight"><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.978800</td><td>0.000001</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.978800</td><td>0.014204</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - softmax</td><td>0.977700</td><td>0.039045</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - softmax</td><td>0.977000</td><td>0.008155</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - softmax</td><td>0.976600</td><td>0.014674</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - softmax</td><td>0.976300</td><td>0.037918</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - softmax</td><td>0.975900</td><td>0.005204</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.973200</td><td>0.019572</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.973100</td><td>0.016679</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.972900</td><td>0.000000</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.972900</td><td>0.022284</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.972300</td><td>0.010000</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - softmax</td><td>0.971500</td><td>0.024638</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - d0.50 - softmax</td><td>0.967700</td><td>0.009937</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - softmax</td><td>0.965400</td><td>0.014436</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.965300</td><td>0.023687</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - softmax</td><td>0.964000</td><td>0.030046</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - softmax</td><td>0.960600</td><td>0.017570</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.960400</td><td>0.010072</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.958800</td><td>0.029385</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.958100</td><td>0.000000</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - softmax</td><td>0.952900</td><td>0.039034</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.950900</td><td>0.023145</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.938000</td><td>0.030095</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.936800</td><td>0.021590</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - softmax</td><td>0.936000</td><td>0.010862</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.935200</td><td>0.029854</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.933000</td><td>0.020006</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - d0.70 - softmax</td><td>0.929400</td><td>0.033064</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.923500</td><td>0.032460</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - softmax</td><td>0.923200</td><td>0.020825</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.921400</td><td>0.050000</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - softmax</td><td>0.920900</td><td>0.003904</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.920300</td><td>0.010648</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - softmax</td><td>0.898600</td><td>0.060130</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.898200</td><td>0.070000</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - softmax</td><td>0.887600</td><td>0.030911</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.886300</td><td>0.050008</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.863900</td><td>0.080204</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.857100</td><td>0.040982</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.850100</td><td>0.050000</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - softmax</td><td>0.829700</td><td>0.090200</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.827300</td><td>0.068081</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.825500</td><td>0.055541</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - softmax</td><td>0.824500</td><td>0.027459</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.776600</td><td>0.053677</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.776200</td><td>0.057055</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.749500</td><td>0.119287</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.690500</td><td>0.064008</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.171600</td><td>0.107028</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.118800</td><td>0.115237</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - d0.90 - softmax</td><td>0.113500</td><td>0.120973</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.112400</td><td>0.100520</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.098200</td><td>0.110445</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.083900</td><td>0.123952</td></tr>
  </tbody>
</table>

#### By avg target score:

<table class="results">
  <thead>
    <tr>
      <th>Model</th>
      <th>Test set accuracy</th>
      <th>Avg target score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.958100</td><td>0.000000</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.972900</td><td>0.000000</td></tr>
    <tr class="highlight"><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.978800</td><td>0.000001</td></tr>
    <tr class="highlight"><td>input - d0.30 - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991200</td><td>0.000005</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.989400</td><td>0.000027</td></tr>
    <tr class="highlight"><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.990300</td><td>0.000588</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.000589</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.989500</td><td>0.000731</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - d0.10 - softmax</td><td>0.991200</td><td>0.000821</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.985800</td><td>0.001278</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - softmax</td><td>0.986200</td><td>0.001474</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - fc - d0.30 - softmax</td><td>0.986100</td><td>0.002538</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.991100</td><td>0.002841</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - softmax</td><td>0.920900</td><td>0.003904</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - softmax</td><td>0.975900</td><td>0.005204</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.005734</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - softmax</td><td>0.981000</td><td>0.006804</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - softmax</td><td>0.977000</td><td>0.008155</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.991400</td><td>0.008776</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - fc - softmax</td><td>0.989400</td><td>0.009515</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - fc - d0.50 - softmax</td><td>0.967700</td><td>0.009937</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.992300</td><td>0.009995</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - softmax</td><td>0.984700</td><td>0.009997</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.972300</td><td>0.010000</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.986900</td><td>0.010000</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - fc - d0.50 - softmax</td><td>0.985100</td><td>0.010018</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.985800</td><td>0.010068</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.960400</td><td>0.010072</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - fc - softmax</td><td>0.980100</td><td>0.010240</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.990700</td><td>0.010305</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.920300</td><td>0.010648</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - softmax</td><td>0.991400</td><td>0.010776</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - softmax</td><td>0.936000</td><td>0.010862</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - softmax</td><td>0.992600</td><td>0.012553</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - fc - softmax</td><td>0.991900</td><td>0.013715</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.013762</td></tr>
    <tr><td>input - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.013994</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.978800</td><td>0.014204</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - softmax</td><td>0.965400</td><td>0.014436</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - softmax</td><td>0.976600</td><td>0.014674</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - softmax</td><td>0.986100</td><td>0.014919</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - d0.10 - softmax</td><td>0.992000</td><td>0.015429</td></tr>
    <tr><td>input - d0.30 - conv1 - d0.30 - conv2 - d0.30 - fc - d0.30 - softmax</td><td>0.973100</td><td>0.016679</td></tr>
    <tr><td>input - conv1 - conv2 - fc - d0.30 - softmax</td><td>0.991700</td><td>0.016735</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - softmax</td><td>0.960600</td><td>0.017570</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - softmax</td><td>0.989700</td><td>0.018690</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.979000</td><td>0.019071</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - softmax</td><td>0.985200</td><td>0.019258</td></tr>
    <tr><td>input - d0.10 - conv1 - d0.10 - conv2 - fc - d0.10 - softmax</td><td>0.991900</td><td>0.019433</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - fc - d0.70 - softmax</td><td>0.973200</td><td>0.019572</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.933000</td><td>0.020006</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - softmax</td><td>0.923200</td><td>0.020825</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.936800</td><td>0.021590</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - fc - softmax</td><td>0.988400</td><td>0.022035</td></tr>
    <tr><td>input - conv1 - d0.10 - conv2 - d0.10 - softmax</td><td>0.972900</td><td>0.022284</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.991700</td><td>0.022410</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - fc - softmax</td><td>0.950900</td><td>0.023145</td></tr>
    <tr><td>input - d0.50 - conv1 - conv2 - d0.50 - fc - softmax</td><td>0.965300</td><td>0.023687</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - softmax</td><td>0.971500</td><td>0.024638</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - fc - softmax</td><td>0.992400</td><td>0.027068</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - softmax</td><td>0.824500</td><td>0.027459</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - fc - d0.90 - softmax</td><td>0.958800</td><td>0.029385</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - softmax</td><td>0.935200</td><td>0.029854</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - d0.10 - fc - softmax</td><td>0.992300</td><td>0.029956</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - softmax</td><td>0.964000</td><td>0.030046</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - fc - softmax</td><td>0.938000</td><td>0.030095</td></tr>
    <tr><td>input - conv1 - conv2 - d0.30 - fc - softmax</td><td>0.979800</td><td>0.030138</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - softmax</td><td>0.887600</td><td>0.030911</td></tr>
    <tr><td>input - d0.70 - conv1 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.923500</td><td>0.032460</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - fc - d0.70 - softmax</td><td>0.929400</td><td>0.033064</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - fc - softmax</td><td>0.992100</td><td>0.033148</td></tr>
    <tr><td>input - conv1 - conv2 - d0.10 - softmax</td><td>0.986600</td><td>0.037620</td></tr>
    <tr><td>input - d0.30 - conv1 - conv2 - d0.30 - softmax</td><td>0.976300</td><td>0.037918</td></tr>
    <tr><td>input - conv1 - conv2 - d0.50 - softmax</td><td>0.952900</td><td>0.039034</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - softmax</td><td>0.977700</td><td>0.039045</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.857100</td><td>0.040982</td></tr>
    <tr><td>input - d0.10 - conv1 - conv2 - softmax</td><td>0.987700</td><td>0.049586</td></tr>
    <tr><td>input - d0.50 - conv1 - d0.50 - conv2 - d0.50 - fc - d0.50 - softmax</td><td>0.921400</td><td>0.050000</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.850100</td><td>0.050000</td></tr>
    <tr><td>input - conv1 - d0.50 - conv2 - d0.50 - softmax</td><td>0.886300</td><td>0.050008</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.776600</td><td>0.053677</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.825500</td><td>0.055541</td></tr>
    <tr><td>input - d0.90 - conv1 - conv2 - d0.90 - fc - softmax</td><td>0.776200</td><td>0.057055</td></tr>
    <tr><td>input - conv1 - conv2 - d0.90 - softmax</td><td>0.898600</td><td>0.060130</td></tr>
    <tr><td>input - conv1 - conv2 - softmax</td><td>0.988900</td><td>0.063591</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.690500</td><td>0.064008</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.827300</td><td>0.068081</td></tr>
    <tr><td>input - conv1 - d0.90 - conv2 - d0.90 - softmax</td><td>0.898200</td><td>0.070000</td></tr>
    <tr><td>input - conv1 - d0.30 - conv2 - d0.30 - softmax</td><td>0.863900</td><td>0.080204</td></tr>
    <tr><td>input - conv1 - conv2 - d0.70 - softmax</td><td>0.829700</td><td>0.090200</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - softmax</td><td>0.112400</td><td>0.100520</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - d0.70 - softmax</td><td>0.171600</td><td>0.107028</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - d0.90 - softmax</td><td>0.098200</td><td>0.110445</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - d0.90 - fc - softmax</td><td>0.118800</td><td>0.115237</td></tr>
    <tr><td>input - conv1 - d0.70 - conv2 - d0.70 - softmax</td><td>0.749500</td><td>0.119287</td></tr>
    <tr><td>input - d0.90 - conv1 - d0.90 - conv2 - fc - d0.90 - softmax</td><td>0.113500</td><td>0.120973</td></tr>
    <tr><td>input - d0.70 - conv1 - d0.70 - conv2 - d0.70 - fc - softmax</td><td>0.083900</td><td>0.123952</td></tr>
  </tbody>
</table>