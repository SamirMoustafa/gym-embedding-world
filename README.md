
OpenAI gym Embedding world Demo
===============================
[![Build Status](https://travis-ci.org/SamirMoustafa/embedding_world.svg?branch=master)](https://travis-ci.org/SamirMoustafa/embedding_world)

Implementation of N-dimension worlds environments for word embedding
from book [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
compatible with [OpenAI gym](https://github.com/openai/gym>).

install environment
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

There are `(2*N)+2` actions in each state {dimension(i)+1, dimension(i)-1} for every i in range from 1 to N and {pickup, dropdown}


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
    howpublished = {\url{https://github.com/SamirMoustafa/embedding_world}},
}
```

