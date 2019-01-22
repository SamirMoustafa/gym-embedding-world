# https://travis-ci.org/SamirMoustafa/embedding_world
from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='gym-embedding_world',
      version='0.0.3.3',
      description='Two word embedding mapping compatible with OpenAI gym.',
      long_description=long_description,
      license="MIT",
      install_requires=['gym', 'numpy', 'gensim', 'atari-py'],
      keywords='embedding_world',
      author_email='samir.moustafa.97@gmail.com',
      include_package_data=True,
      author='Samir Moustafa',
      packages=find_packages(),
      python_requires='>=3.5,<3.7',
      url='https://github.com/SamirMoustafa/gym-embedding-world/',
      package_data={'gym-embedding_world': ['embedding_world/envs/world_sample//*.vec']},
      scripts = [ 'embedding_world/__init__.py',
                  'embedding_world/envs/__init__.py',
                  'embedding_world/envs/embedding_world_env.py',
                  'embedding_world/envs/embedding_world_handler.py'],
      classifiers=[
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3 ",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
      ]
      )
