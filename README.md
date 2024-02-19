## What it is all about

Typical problem while applying the autencoders (AE) is connected with the fact that we are looking not just for all possible outliers which our AE can deliver, but only for some *specific ones* defined according to some particular criteria, e.g. set by a customer. Let's call such outliers "defects". E.g., by applying an AE within the production line which produces defects rarely it can happen that the other types of outliers will be estimated by an AE stronger than the defects.

Without applying some specific metrics your AE won't be able to distinguish them from the rest of outliers. In this case it appears very helpful to make use of some prior knowledge about this specific type of outliers, if available. In real life a handful of the defect representatives can be always collected. The basic idea then is to tell the AE, what type of elements it *should not learn*. In the context of the model fitting it can be realized by introducing some extra loss penalty for such elements, which is called "contrastive" loss. Then the total loss will be the sum of the "traditional" and "contrastive" contributions:

$${\rm loss} = \sum_{i}\left(x_{i} - {\rm model}(x_{i})\right)^2 - \sum_{i,d}\left({\rm model}(x_{i})- {\rm model}(x_{d})\right)^2 + \sum_{i>j}\left({\rm model}(x_{i})-{\rm model}(x_{j})\right)^2,$$
where the "traditional" recovery loss is represented by the first term and the last two terms represent the "contrastive" contribution. Whereas the second (negative) term maximizes the distance between the main group ($x_{i}$) and the defect ($x_{d}$) clouds, the third (positive) term tries to compact the main group points together,  mainly serving as a supplemental for the second one.

If the contrastive loss will be literally implemented as it is shown here, its backpropagation through the whole model will affect both decoder and encoder directly and might strongly spoil the converency of the recovery term. For this reason it is adviced to compute contrastive part of the loss within the latent space. By reorganizing the latent space should not dramatically affect the encoder/decoder functions connecting it with the "real" space. Taking this into account the previous expression turns to
$${\rm loss} = \sum_{i}\left(x_{i} - {\rm model}(x_{i})\right)^2 - \sum_{i,d}\left({\rm enc}(x_{i})- {\rm enc}(x_{d})\right)^2 + \sum_{i>j}\left({\rm enc}(x_{i})-{\rm enc}(x_{j})\right)^2,$$
where $\rm enc$ - is the encoder component of the model (i.e., the whole model is $${\rm model(x)} = dec(enc(x))$$)


