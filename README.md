# Textual Anomaly Detection for Goal-Oriented Conversational Models

Research on methods of anomaly detection for chatbots which was presented in a poster format on *Russian Summer School of Information Retrieval 2017* (RuSSIR'17). The .pdf version of the poster could be found here: https://yadi.sk/i/UEi2jwU43MFWVi

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

If you have any questions or comments on the paper please feel free to contact me:

amir{my username on GitHub}@gmail.com
