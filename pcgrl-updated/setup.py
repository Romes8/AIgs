from setuptools import setup, find_packages

# Read the README file to include in the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gym_pcgrl',  # The name of your package
    version='0.4.0',  # Version of the package (you can update this as needed)
    install_requires=['gymnasium', 'numpy>=1.17', 'pillow'],
    author="Ahmed Khalifa",  # Keep original author information
    author_email="ahmed@akhalifa.com",  # Keep original author email
    description="A package for Procedural Content Generation via Reinforcement Learning OpenAI Gym interface.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amidos2006/gym-pcgrl",  # Original repository URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),  # Automatically find all subpackages
    include_package_data=True,  # Include any other files specified in MANIFEST.in (e.g., README.md)
    python_requires='>=3.6',  # Make sure your package is Python 3.6 or later
)