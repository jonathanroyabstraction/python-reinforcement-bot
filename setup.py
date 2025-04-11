from setuptools import setup, find_packages

setup(
    name="wow-bot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22.0",
        "opencv-python>=4.5.5.64",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "jsonschema>=4.17.3",
        "mss>=6.1.0",
        "pynput>=1.7.6",
        "pyobjc-core>=8.5.1",
        "pyobjc-framework-Quartz>=8.5.1",
        "fastapi>=0.85.0",
        "uvicorn>=0.18.3",
        "torch>=1.11.0",
        "torchvision>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "wow-bot=src.core.main:main",
        ],
    },
)
