# Textual Anomaly Detection for Goal-Oriented Conversational Models

Dependencies:

* numpy
* pandas
* gensim
* scikit_learn
* tensorflow
* keras
* pymorphy2
* matplotlib
* seaborn

Code of the experiments is proposed at `scikit_learn_anomaly.ipynb`

You should create a directory `models` and put there [Google News model](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) (rename it to `google_news.bin`) and [Dvach model](https://yadi.sk/d/QzNiFKTY3M3odW). Then download datasets [`2ch-topics`](https://yadi.sk/d/3FukjFEs3M3ogz) and [`reddit-topics`](https://yadi.sk/d/2AaZNNFy3M3ohP) and put them in the root of the project.

Then you will be able to reproduce the experiments.
