# Mixture-of-Softmaxes
This is the backend repository of the medium article "Mixture-of-Softmaxes for Deep Session-based Recommender Systems".

## TL;DR:
The traditional softmax is limited in its capacity to fully model tasks like natural language that are highly context-dependent. This limit in expressiveness called the Softmax bottleneck can be described through the lenses of matrix factorization and the study of the resulting matrix ranks.
The Mixture of Softmaxes by Yang et al. allows for overcoming the bottleneck by increasing the expressiveness of the network without exploding the number of parameters.
In this work, we study whether this limitation also applies to deep session-based recommender systems using the movielens-25m dataset.


## Installation
To install the requirements you can use conda. You can then simply type 
`make install` or `make install_m2` if you are on a mac m2 or m1.


## Conclusion
The main takeaways from this work are the following. The MoS effectiveness is correlated with:
- data volume. It introduces an additional burden on the data volume requirements per item by increasing modeling complexity. Performance is increased when the number of observations per item is sufficient.
- diversity in user sequences. It is only effective if the complexity of the session-based dataset requires it.

MoS is more effective on low-dimensional embeddings than on high-dimensional ones.

Final note: Although the improvements are pretty small for this dataset. I still think improvements might be much more important for some industrial datasets having very long sequences and a lot of diversity.