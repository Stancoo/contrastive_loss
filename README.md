## What it is all about

Typical problem while applying the autencoders (AE) is connected with the fact that we are looking not just for all possible outliers which our AE can deliver, but only for some *specific ones* defined according to some particular criteria, e.g. set by a client. Let's call such outliers "defects". E.g., by applying an AE within the production line which produces defects rarely it can happen that the other types of outliers will be estimated by an AE stronger than the defects.

Without applying some specific metrics your AE won't be able to distinguish them from the rest of outliers. In this case it appears very helpful to make use of some prior knowledge about this specific type of outliers, if available. In real life a handful of the defect representatives can be always collected. The basic idea then is to tell the AE, what type of elements it *should not learn*. In the context of the model fitting it can be realized by introducing some extra loss penalty for those elements: `-(x-x')**2`, where `x` is the input and `x'=model(x)` is the output of the AE model. Minus sign takes care of the fact that the difference `(x-x')**2` gets maximized if the element represents a defect. The total loss represents a sum of the "traditional" contribution and the "contrastive" addon:

$`loss = \sum_{i}(x_{i}-model(x_{i}))^2 - \sum_{i,d}(model(x_{i})-model(x_{d}))^2 + \sum_{i,j}(model(x_{i})-model(x_{j}))^2`$,

where the "traditional" loss is represented by the first term and the last two terms are "contrastive" contributions. Whereas the second (negative) term maximizes the distance between the main group ($x_{i}$) and the defect ($x_{d}$) clouds; the third one (positive term) tries to compact the main group points together.

