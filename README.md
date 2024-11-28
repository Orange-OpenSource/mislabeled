# Model-probing mislabeled examples detection in machine learning datasets

EN. Detect mislabeled examples in machine learning dataset, using the 4 components framework described in the paper [Mislabeled examples detection viewed as probing machine learning models: concepts, survey and extensive benchmark](https://openreview.net/forum?id=3YlOr7BHkx), which allows the implementation of a variety of model-probing detection methods.

FR. Détection d'exemples mal-étiquetés dans des jeux de données d'apprentissage automatique, en utilisant les 4 composants décrits dans l'article [Mislabeled examples detection viewed as probing machine learning models: concepts, survey and extensive benchmark](https://openreview.net/forum?id=3YlOr7BHkx), qui permet d'implémenter une multitude de méthodes de détection par sondage de modèle.

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