# Attentive Collaborative Filtering

This is a repository for a pytorch cover of ACF recommender from paper [Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-AttentiveCF.pdf).

This is **not** a direct model from paper but rather an interpretation of it. You can see a list of differences lower.

## Dataset
I use Movielens dataset. It will be automatically downloaded. Just pass a name

- ml-20m
- ml-latest-small
- ml-latest

See [dataset site](https://grouplens.org/datasets/movielens/) for details on datasets and other archive names to grab.
```python
from dataset_handler import MovieLens
ml = MovieLens('ml-latest-small')
```

## Training process
Training is handled in a triple loss style with an explicit warp as a loss function.
For each pair (user, positive rating) we sample a batch of negative (not seen) examples for that user.
Then we make sure that score for a positive item is bigger than score for negative items according to ewarp loss.

## Nets
There are 2 nets: **FeatureNet** to account for component level attention and 
**UserNet** to account for item level attention and provide final user embedding.

#### FeatureNet
Takes user embedding and features for history items for this user. 
Returns a vector of components for this items, 
considering this user's attention to item components.

#### UserNet
Takes user id and ids for history items for this user.
Returns user embedding considering item level attention.

Prediction for user embedding and item embeddings can be retrieved by method `score` of UserNet.


##### Todo: 

- Send model to device for GPU support



### Differences

- Using Kaiming initialization for networks instead of Xavier because of ReLU activations. Network is shallow so wrong initialization from paper shouldn't really hurt, but still.
- In paper component vector is meant to be embedded into the same space as user. I treat component vector as an arbitrary feature vector of any size, not actualy as independent component. So I don't use component attention since there is only one component.
- I don't use auxiliary item vector <img src="https://render.githubusercontent.com/render/math?math=p"> because can't find any reason to.
- My version of warp loss is different from the normal one. Instead of sampling negative items until violation is met to estimate rank I sample a batch of negatives and weight each violating example by <img src="https://render.githubusercontent.com/render/math?math=w=ln(\sum_i^{|batch|} v_i %2B 1)%2B1"> where <img src="https://render.githubusercontent.com/render/math?math=v_i=1"> if <img src="https://render.githubusercontent.com/render/math?math=i\text{-th}"> example is violated. Thus I eliminate the need to sample iteratively. If I've used position of first violation it would be the same as a regular warp loss with a limit of draws set to batch size.
- Original warp method is for implicit feedback, but movielens has ratings so I additionaly weighted loss with <img src="https://render.githubusercontent.com/render/math?math=w = \frac{r_{positive} %2B (r_{max} - r_{negative})}{r_{max}}"> to account for explicitness. If the item was not observed and we don't know <img src="https://render.githubusercontent.com/render/math?math=r_{negative}"> , it is set to <img src="https://render.githubusercontent.com/render/math?math=r_{unknown}"> which is configurable and is meant to be a mean rating. It may be changed to <img src="https://render.githubusercontent.com/render/math?math=\bar{r}_{items}"> or <img src="https://render.githubusercontent.com/render/math?math=\bar{r}_{user}"> but is not implemented atm. See `ewarp_loss` (explicit warp) for details. 



