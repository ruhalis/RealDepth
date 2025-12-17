from setuptools import setup, find_packages


setup(
    name="realdepth",
    version="0.0.1",
    description="Real time 2D camera depth estimation",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8,<3.11",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "pillow",
        "pyyaml",
        "tqdm",
        "tensorboard",
        "matplotlib"
        # pyrealsense2
    ],
    extras_require={
        "realsense": ["pyrealsense2"],
    },
)


