## What it is all about

Typical problem while applying the <a href="https://en.wikipedia.org/wiki/Autoencoder">autoencoders</a>  (AE) is connected with the fact that we are looking not just for all possible outliers which our AE can deliver, but only for some *specific ones* defined according to some particular criteria, e.g. set by a customer. Let's call such outliers "defects". E.g., by applying an AE within the production line which produces defects rarely it can happen that the other types of outliers will be estimated by an AE even stronger than the defects:

<p align="center"> <img title="Upper panel: schematic distribution of elements depicted as points. The defects are shown as red. Lower panel: schematic loss distribution delivered by an AE-model. Vertical line approximately separates outliers from the main group. Red rods correspond to defects." alt="Fig." src="/pics/fig1.svg" width=25% style="display: block; margin: auto" >

Without applying some specific metrics your AE won't be able to distinguish them from the rest of outliers. In this case it appears very helpful to make use of some prior knowledge about this specific type of outliers, if available. In real life a handful of the defect representatives can be always collected. The basic idea then is to tell the AE, what type of elements it *should not learn*. In the context of the model fitting it can be realized by introducing some extra penalty loss for such elements, which is called <a href="https://arxiv.org/abs/2007.11457">"contrastive" loss</a>  . Then the total loss will be the sum of the "traditional" and "contrastive" contributions (squared difference loss is just a particular case):

$${\rm loss} = \sum_{i}\left(x_{i} - {\rm model}(x_{i})\right)^2 - \sum_{i,d}\left({\rm model}(x_{i})- {\rm model}(x_{d})\right)^2 + \sum_{i>j}\left({\rm model}(x_{i})-{\rm model}(x_{j})\right)^2,$$
where the "traditional" recovery loss is represented by the first term and the last two terms represent the "contrastive" contribution. Whereas the second (negative) term maximizes the distance between the main group ($x_{i}$) and the defect ($x_{d}$) clouds, the third (positive) term tries to compact the main group points together,  mainly serving as a supplemental for the second one. Both contrastive loss contributios are shown below schematically as a sum of the black and the red connectors. The arrow indicates what kind of data transformation (i.e., separation of defects) is assumed to achieve by adding contrastive loss to the fitting:

<p align="center"> <img title="Schematic representation of the contrastive loss contributions. The defects are shown as red. The aim is to minimize the sum of the distances between the main-group elements (black connectors) and to maximize the sum of the distances between main-group  and the defects (red connectors)." alt="Fig." src="/pics/fig2.svg" width=50% style="display: block; margin: auto" >


If the contrastive loss would be literally implemented as it is shown here, its backpropagation through the whole model will affect both decoder and encoder directly and might strongly spoil the converency of the recovery term. For this reason it is adviced to compute contrastive part of the loss within the latent space. Reorganizing the latent space should not dramatically affect the encoder/decoder functions connecting it with the "real" space. Taking this into account the previous expression turns to
$${\rm loss} = \sum_{i}\left(x_{i} - {\rm model}(x_{i})\right)^2 - \sum_{i,d}\left({\rm enc}(x_{i})- {\rm enc}(x_{d})\right)^2 + \sum_{i>j}\left({\rm enc}(x_{i})-{\rm enc}(x_{j})\right)^2,$$
where $\rm enc$ - is the encoder component of the model. The whole model is the combination of both, encoder and decoder: ${\rm model}(x) = {\rm dec}({\rm enc}(x)).$


## Current repo

To demonstrate the advantage of the contrastive loss account on a particular example,  as a model we take a version of the so-called <a href="https://link.springer.com/article/10.1007/s10489-022-03613-1">Spatio-Temporal AE</a> translated into Pytorch. It is intended to classify the video fragments in a binary sense, i.e. normal/anomal. Training and testing procedures are configured in `config.json`:

```
{
    "n_channels": 1,
    "path_save_model": "model_sav",
    "path_scratch": "scratch",
    "batch_size_train": 100,
    "batch_size_val": 8,
    "seqlen": 21,
    "height": 101,
    "width": 101,
    "learning_rate": 0.0005,
    "epochs": 20000,
    "loss_type": "mse",
    "dtheta": 2.0,
    "contrastive_loss":{
	"n_outliers": 8,
	"n_outliers_val":2,
	"dtheta": 2.1
    },
    "inference": {
	"results": "res_dtheta=1.9",
	"weights": "model_sav/best.pth",
	"loss_type": "mse",
	"n_samples": 1000,
	"random_seq": "False"
    }
}

```


As an input the model takes a sequence of frames of the fixed length, defined in `config.json` as `"seq_len":21`. The sequences will be all formed by replicating the same "coin" image `img.png` which is  rotated by 2 degrees from frame to frame:
<p align="center"> <img title="Sequence of 21 frames, used as an input. Each frame is obtained by rotating the previous one by 2 degrees." alt="Fig." src="/pics/fig3_sequence.png" width=80% style="display: block; margin: auto" >

In `config.json` the corresponding entry is `"d_theta":2.0`. Since the start angle is arbitrary, we can generate such sequences infinitely. For this reason, one epoch consists of a single batch. One training batch contains `"batch_size_train": 100` sequences. Each frame will have the size "height"$\times$"width", both defined in `config.json`