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

### 2. Compute Representer values of the MLP

```python
probe = Representer()
representer_values = probe(mlp, X, y)
```

### 3. Inspect your training data

```python
supicious = np.argsort(-representer_values)[0:top_k]
for i in suspicious:
  plt.imshow(X[i].reshape(28, 28))
```

### 4. Wanna get the variance of the Representer values during training ?

```python
detector = ModelProbingDetector(mlp, Representer(), ProgressiveEnsemble(), "var")
var_representer_values = detector.trust_scores(X, y)
```

## Predefined detectors

| Detector | Paper | Code (`from mislabeled.detect.detectors`) |
| - | - | - |
| Area Under the Margin (AUM) | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf) | `import AreaUnderMargin` |
| Influence | [Paper 1974](https://www.tandfonline.com/doi/abs/10.1080/01621459.1974.10482962) | `import InfluenceDetector` |
| Cook's Distance | [Paper 1977](https://www.jstor.org/stable/1268249) | `import CookDistanceDetector` |
| Approximate Leave-One-Out | [Paper 1981](https://www.jstor.org/stable/2240841) | `import ApproximateLOODetector` |
| Representer | [Paper 1972](https://www.jstor.org/stable/2240067) | `import RepresenterDetector` |
| TracIn | [NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf) | `import TracIn` |
| Forget Scores | [ICLR 2019](https://openreview.net/pdf?id=BJlxm30cKm) | `import ForgetScores` |
| VoG | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Agarwal_Estimating_Example_Difficulty_Using_Variance_of_Gradients_CVPR_2022_paper.pdf) | `import FiniteDiffVoG, FiniteDiffVoLG, VoLG`|
| Small Loss | [ICML 2018](https://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf) | `import SmallLoss`|
| CleanLab | [JAIR 2021](https://www.jair.org/index.php/jair/article/view/12125/26676) | `import ConfidentLearning` |
| Consensus (C-Scores) | [Applied Intelligence 2011](https://link.springer.com/article/10.1007/s10489-010-0225-4) | `import ConsensusConsistency`|
| AGRA | [ECML 2023](https://dl.acm.org/doi/10.1007/978-3-031-43412-9_14) | `import AGRA` |

and other limitless combinations by using `ModelProbingDetector` with any `probe` and `Ensembles` from the library.

Most of these detectors work for both regression and classification diagnostics.

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

Formatting and linting is done with ruff as a [pre-commit](https://pre-commit.com/):
- install: ```pre-commit install```, 
- format and lint: ```pre-commit run --all-files``` (automatically done before a commit).

Run tests with [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer): ```uv run pytest```.