# Uncertainty Quantification in CNN Through the Bootstrap of Convex Neural Networks

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

This work aims to reproduce the results of and further provide an alternative implementation for [Uncertainty Quantification in CNN Through the Bootstrap of Convex Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17434) by Hongfei Du, Emre Barut and Fang Jin [1].

## 1.1. Paper summary

Uncertainty quantification is an important task for almost any critical application that neural networks are in use. Despite that, the literature regarding it is freshly developing. There are many novel methods aiming to quantify uncertainty, however most of them do not have the theoretical guarantees regarding the quality of the quantification process. This [work](https://ojs.aaai.org/index.php/AAAI/article/view/17434) aims to provide a cheaper and theoretically supported alternative to novel Bayesian alternatives through prospoing a novel bootstrap based framework for the predictive uncertainty estimation. The inference procedure of the framework relies on convexified convolutional neural networks, [CCNNs](https://arxiv.org/abs/1609.01000) [2]. They further apply a "warm start" approach to boostrapping mechanism to reduce the number of sufficient iterations by a sizable amount. The authors finally propose a novel transfer learning approach to further increase the usability of their work.


# 2. The method and our interpretation

## 2.1. The original method

The method enjoys two nice properties coming from the CCNNs and warm-start bootstrapping:

* Firstly, he convexity of CCNNs both guarantee the statistical validity and theoretical background for the method and also the global optimum for the subsampled dataset.

* Secondly, the quantification process is quick since the method utilizes a warm-strat approach during bootstrapping, allowing it to initialize itself through the parameters of the previous solution.

The original method that the authors propose is summarized neatly on the following Figure 1:

![bootstrap-algorithm](readme_assets/bootstrap-algorithm.png "Figure 1")
<figcaption align="center">Figure 1 - CCNN Bootstrap Algorithm</figcaption>

Verbally, the method work as:

* First, train the network in order to have the weights for the first bootstrap sampling's warm-start
* Secondly, start the bootstrap sampling by initializing the model with previously saved parameters
* Then obtain some subset of dataset to train the network for that particular bootstrap
* After each sampling iteration, save the weights of the previous model for future use and save the predictions for statistics
* Finally, output the predictions of the model with their intervals

The authors further propose a novel transfer learning method, which is to utilize a backbone that was trained for a similar task before (i.e VGG16 pretrained on ImageNet) and add a CCNN layer right after the last convolutional layer of it. It is worth noting that if the backbone's training data somehow has intersections with the bootstrapped dataset, then the theoretical validity of this method would become invalid. In case of an inavailability of such pretrained networks, the authors further propose the following three techniques to obtain similar pretrained networks:
* Train & Forget: Start training a CNN on a certain dataset, after a certain number of epochs replace the dataset with another one, train until model outputs almost random guesses
* Train & Flip: Start training a CNN with the original labels of a dataset, then randomly flip labels at a certain time and continue training until the network overfits
* Train & Perturb: After training a CNN, add random perturbations to its weights


The authors choose the infamous [Deep Ensembles](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)[3] method and a shallow CNN, [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)[4] as baselines to compare their methods against. In order to do so, they consider the following metrics:

* Average length of the 95% confidence interval of the predictions of the model, shorter interval implying lower uncertainty
* Average log-likelihood, i.e the average cross-entropy by each bootstrap sampling iteration, or more formally:
 $$ L = \frac{1}{B} \sum_{b=1}^B \sum_{i=1}^N H(p_i^b, y_i)$$
 
## 2.2. Our interpretation 

We had some problems with the CCNN implementation. The design for it was done in 2017 and it was implemented without any framework support. (KIVANÇ BURAYA HER NE SORUN YAŞADIYSAN YAZ). Furthermore, we could not obtain the results for the [Deep Ensembles](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)[3] method as it is an expensive method for us to run on Google Colaboratory machines.

<CCNN dataset usage assumptions? batch size, vectorization etc.?>

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

We have omitted <...> step in input normalization due to an issue...

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

@TODO: Update/recreate environment.yaml when releasing with `conda env export > environment.yaml`

We have used Python 3.9 with the following packages to run the code:

* numexpr
* numpy
* pytorch
* scikit-learn
* torchvision

To install these packages with Conda, run

```conda create --name ccnn-bstrap-uq scikit-learn numpy numexpr```

Then install pytorch by using the desired command from [their website](https://pytorch.org/get-started/locally/).
We have used CUDA 10.2:

```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```


If desired, our environment can be cloned using Conda and the `environment.yaml` file using the following command:

```conda env create -f environment.yaml```

To activate:

```conda activate ccnn-bstrap-uq```

<Run `<...>.py` to replicate the <...> experiment...>:

```python3 <...>.py```

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

![paper-results](readme_assets/<...>.png "Figure 2")
<figcaption align="center">Figure 2 - Results of <...> from the paper</figcaption>

<...discussion...>

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

`main/_init_paths` adapted from bag of tricks...

CCNN and related math func.s adapted from [zhang](https://github.com/zhangyuc/CCNN/blob/master/src/mnist/CCNN.py) ...


# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.

Selim Kuzucu:
