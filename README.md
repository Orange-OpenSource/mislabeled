# mislabeled

> Model-probing mislabeled examples detection in machine learning datasets

A `ModelProbingDetector` assigns `trust_scores` to training examples $(x, y)$ from a dataset by `probing` an `Ensemble` of machine learning `model`.

## Install

```console
pip install git+https://github.com/orange-opensource/mislabeled
```

## Find suspicious digits in MNIST

### 1. Train a MLP on MNIST

```python
X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
y = LabelEncoder().fit_transform(y)
mlp = make_pipeline(MinMaxScaler(), MLPClassifier())
mlp.fit(X, y)
```

### 2. Compute Self Representer values of the MLP

```python
probe = Representer()
self_representer_values = probe(mlp, X, y)
```

### 3. Inspect your training data

```python
supicious = np.argsort(-self_representer_values)[0:top_k]
for i in suspicious:
  plt.imshow(X[i].reshape(28, 28))
```

### 4. Wanna get the variance of the Self Representer values during training ?

```python
detector = ModelProbingDetector(mlp, Representer(), ProgressiveEnsemble(), "var")
var_self_representer_values = detector.trust_scores(X, y)
```

## Tutorials

For more details and examples, check the [notebooks](https://github.com/orange-opensource/mislabeled/tree/master/examples) !

## Paper

If you use this library in a research project, please consider citing the corresponding paper with the following bibtex entry:

    @article{george2024mislabeled,
      title={Mislabeled examples detection viewed as probing machine learning models: concepts, survey and extensive benchmark},
      author={Thomas George and Pierre Nodet and Alexis Bondu and Vincent Lemaire},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2024},
      url={https://openreview.net/forum?id=3YlOr7BHkx},
      note={}
    }

## Development

Install [hatch](#https://hatch.pypa.io/latest/install/).

To format and lint:
```console
hatch fmt
```

To run tests:
```console
hatch test
```