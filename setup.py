from setuptools import setup

setup(name='sghmc',
      version='0.1',
      description='An implementation of Stochastic Gradient Hamiltonian Monte Carlo by Gilad Amitai and Beau Coker.',
      license='MIT',
      package=['sghmc'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)