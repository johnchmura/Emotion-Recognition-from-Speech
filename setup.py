from setuptools import setup, find_packages

setup(
    name="speech_emotion_recognition",
    version="0.1.0",
    author="John Chmura",
    author_email="jchmura@hawk.iit.edu",
    description="HMM‑based speech/emotion and song/speech detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "soundfile",
        "librosa",
        "hmmlearn",
        "scikit-learn",
        "matplotlib",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "ser=speech_emotion_recognition.ser:cli"
        ],
    },
    python_requires=">=3.8",
)