from setuptools import setup, find_packages

setup(
    name="Kosmosius",
    version="0.1.0",
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=[
        "transformers==4.31.0",
        "datasets==2.14.0",
        "nltk==3.8.1",
        "torch==2.4.1",
        "numpy",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "internetarchive",
        "tqdm",
        "PyYAML",
    ],
    entry_points={
        'console_scripts': [
            'preprocess_data = scripts.preprocess_data:main',
            'create_dataset = scripts.create_dataset:main',
            'train_model = scripts.train_model:main',
            'evaluate_model = scripts.evaluate_model:main',
            'generate_samples = scripts.generate_samples:main',
            'run_all = scripts.run_all:main',
            # Add other scripts as needed
        ],
    },
    python_requires='>=3.8',
)
