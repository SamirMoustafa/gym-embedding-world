from setuptools import setup, find_packages

setup(name='embedding_world',
      version='0.0.1',
      description='Embedding world environments for OpenAI gym.',
      author_email='samir.moustafa.97@gmail.com',
      author='Samir Moustafa',
      packages=find_packages(),
      install_requires=['gym','numpy']
)
