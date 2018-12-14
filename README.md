
OpenAI gym Embedding world Demo
===============================
[![Build Status](https://travis-ci.org/SamirMoustafa/embedding_world.svg?branch=master)](https://travis-ci.org/SamirMoustafa/embedding_world)

Implementation of N-dimension worlds environments for word embedding compatible with [OpenAI gym](https://github.com/openai/gym>).

Install environment
-------------------

.. code::

    $ cd gym-gridworld
    $ conda env create -f environment.yml
    $ source gridworld
    $ pip install -e .

Usage
-----

.. code::

        $ python >>> import gym
        $ python >>> import embedding_world
        $ python >>> env = gym.make('embedding_world-v0')

``embedding_world-v0``
----------------------

Embedding world is simple N-dimension world for example the [Stanfrod GloVe](https://nlp.stanford.edu/projects/glove/) or [facebook fastText models](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).

There are `2N + 2` actions `{dimension(i)+1, dimension(i)-1}` ∀ i in range from 1 to N  ∪ ` {pickup, dropdown}`


which deterministically cause the corresponding state transitions
but actions that would take an agent of the grid leave a state unchanged.
The reward is -1 for all tranistion until the goal is reached.
The terminal state(goal) is represent in a vector/s.

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{embedding_world,
    author = {Samir Moustafa},
    title = {Embedding Environment for OpenAI Gym},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/SamirMoustafa/embedding_world}}
}
```

