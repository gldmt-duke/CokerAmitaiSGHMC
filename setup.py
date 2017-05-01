from setuptools import setup

setup(name='sghmc',
      version='0.1',
      description='An implementation of Stochastic Gradient Hamilton Monte Carlo by Gilad Amitai and Beau Coker.',
      license='MIT',
      packages=['vae'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)