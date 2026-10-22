from setuptools import find_packages, setup

setup(
    name="feedback_world_model",
    version="1.0.0",
    description="Feedback World Model Enables Precise Guidance of Diffusion Policy",
    packages=find_packages(include=["fwm", "fwm.*", "dyn_model", "dyn_model.*", "diffusion_policy", "diffusion_policy.*"]),
    python_requires=">=3.9",
)
