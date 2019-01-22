# https://travis-ci.org/SamirMoustafa/embedding_world
from setuptools import setup, find_packages

with open('README.md') as readme_file:
      readme = readme_file.read()

setup(name='gym-embedding_world',
      version='0.0.1',
      description='Two word embedding mapping compatible with OpenAI gym.',
      long_description=readme,
      license="MIT",
      install_requires=['gym', 'numpy', 'gensim', 'atari-py'],
      keywords='embedding_world',
      author_email='samir.moustafa.97@gmail.com',
      include_package_data=True,
      author='Samir Moustafa',
      packages=find_packages(),
      python_requires='>=3.5,<3.8',
      url='https://github.com/SamirMoustafa/gym-embedding-world/',
      package_data={'gym-embedding_world': ['embedding_world/envs/world_sample//*.vec']},
      scripts = [ 'embedding_world/__init__.py',
                  'embedding_world/envs/__init__.py',
                  'embedding_world/envs/embedding_world_env.py',
                  'embedding_world/envs/embedding_world_handler.py'],
      classifiers=[
            "Development Status :: 3 - Alpha",
            "Operating System :: OS Independent",
            "Development Status :: 2 - Pre-Alpha",
            "Programming Language :: Python :: 3 ",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
      ]
      )
