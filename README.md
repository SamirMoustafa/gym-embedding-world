
OpenAI gym Embedding world
==========================

<div align="center">
  <img width="40%" src="https://raw.githubusercontent.com/SamirMoustafa/gym-embedding-world/master/assets/9-dimensional-hypercube.gif"><br><br>
  <h6>An eight-dimensional hypercube graph.</h6>
</div>

[![Build Status](https://travis-ci.org/SamirMoustafa/gym-embedding-world.svg?branch=master)](https://travis-ci.org/SamirMoustafa/gym-embedding-world)

Two word embedding mapping compatible with [OpenAI gym](https://github.com/openai/gym>).

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Gensim

Install environment on anaconda
-------------------------------

    $ conda env create -f gym-embedding-world/environment.yml
    $ source embedding-world
    $ pip install -e gym-embedding-world/.

Install environment on colab
----------------------------

    !git clone "https://github.com/SamirMoustafa/gym-embedding-world.git"
    !pip install -e gym-embedding-world/.
    !mv gym-embedding-world gym-embedding-world-org
    !cp -r gym-embedding-world-org/embedding_world /content
    !ls embedding_world

Usage
-----

        $ python >>> import gym
        $ python >>> import embedding_world
        $ python >>> env = gym.make('embedding_world-v0')
        $ python >>> env.set_paths(embedding_from_file="... YOUR EMBEDDING PATH TO MAP FROM IT  ...",
                                   embedding_to_file  ="... YOUR EMBEDDING PATH TO MAP TO IT  .....")
        $ python >>> env.production_is_off()
        $ python >>> env.set_sentences('... YOUR SENTENCE TO TRANSLATE FROM IT ...', 
                                       '... YOUR SENTENCE TO TRANSLATE TO IT .....')
        $ python >>> state, reward, done, info = env.step('dim(0)+1')

``embedding_world-v0``
----------------------

Embedding world is a simple environment based on OpenAI gym, that loads two-word embedding e.g. [Stanfrod GloVe](https://nlp.stanford.edu/projects/glove/) or [facebook fastText models](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)  with N-dimension and moves from one word(s) embedding-location to the other embedding using an agent actions such that actions that could be taken are `2N + 1` actions `{dimension(i)+1, dimension(i)-1}` ∪ ` {pickup}` ∀ `i` in range from 1 to N

which deterministically cause the corresponding state transitions
but actions that would take an agent of the grid leave a state unchanged.
The reward is negative for all transition until the goal is reached.
The terminal state(goal) is represented in a vector/s.

This environment has been built as part of a graduation project at [University of Alexandria, Department of Computer Science](http://sci.alexu.edu.eg/index.php/en/)

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{embedding_world,
    author = {Samir Moustafa},
    title = {Embedding Environment for OpenAI Gym},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/SamirMoustafa/gym-embedding-world}}
}
```