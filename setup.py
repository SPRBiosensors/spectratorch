from setuptools import setup, find_packages

setup(
    name='spectratorch',
    version='1.0.0',
    description='Workflow pour générer un modèle de classification de qualité des sirops d\'érable',
    install_requires=['pytorch-lightning'],
    packages=find_packages(where="src"),
)

