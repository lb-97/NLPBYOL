# Improving Word Representations Using Self-Supervised Learning

We employed BYOL, a self-supervised implicit contrastive learning strategy, to incorporate grammatical information such as synonyms and antonyms into existing word2vec embeddings. We did this to enhance current word vectors with more information early on in a pipeline as we thought it was vital to leaarn grammar first before anything else like we humans do. We tested our enhanced embeddings on a Sentiment Analysis based classification task on MovieReview dataset and achieved an increase in 2.5% accuracy by using a single GPU for a couple of hours.

To train BYOL from scratch and obtain enhanced embeddings:
```bash
from get_byol_embeddings import get_byol_embed
get_byol_embed()
```
This piece of code will save models at every epoch in current folder

To test our latest embeddings on Movie Review Classification task:
```bash
from get_byol_embeddings import test_embeddings
test_embeddings()
```

To test word2vec embeddings on Movie Review Classification task:
```bash
from get_byol_embeddings import test_embeddings
test_embeddings(enhance=False)
```

To perform PCA and visualize and compare both embeddings for a particular word:
```bash
from get_byol_embeddings import pca
pca(word='get')
```

