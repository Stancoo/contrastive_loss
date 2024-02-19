## What it is all about

Typical problem while applying the autencoders (AE) is connected with the fact that we are looking not just for all possible outliers which our AE can deliver, but only for some *specific ones* defined according to some particular criteria, e.g. set by a client. Let's call such outliers "defects". E.g., by applying an AE within the production line which produces defects rarely it can happen that the other types of outliers will be estimated by an AE stronger than the defects.

Without applying some specific metrics your AE won't be able to distinguish them from the rest of outliers. In this case it appears very helpful to make use of some prior knowledge about this specific type of outliers, if available. In real life a handful of the defect representatives can be always collected. The basic idea then is to tell the AE, what type of elements it *should not learn*. In the context of the model fitting it can be realized by introducing some extra loss penalty for those elements: `loss_contrastive = -(x-x')**2`, where `x` is the input and `x'=model(x)` is the output of the AE model. Minus sign takes care of the fact that the difference `(x-x')**2` get maximized if the element represents a defect.


