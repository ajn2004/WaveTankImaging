# 🌊 Wave Tank Imaging 🌊

A python-enabled workflow to measure peaks and troughs generated in a wave tank for a BioBus Project

## Table of Contents
- [Quick Start](#quick-start)
- [Dependencies](#dependencies)

## Quick Start
To get started with the project from scratch, ensure you have python version [what version?] installed on your computer

Once python is installed, download the git repository and navigate to the downloaded directory
```bash
git clone git@github.com:ajn2004/WaveTankImaging.git
cd WaveTankImaging
```

To keep a clean development system, it's best practice to create a virtual environment and install the dependencies locally with the following commands

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

From there you should be able to run the code with

```bash
pyton main.py
```
## Requirements

Need to take in a video and output amplitude of waves recorded in video
In order to work this code needs to perform the following actions (in order)
- Load Video (mp4) to Image stack
- Determine Pixel size
2) Separate out the frames
3) For each frame
	1) Determine a baseline
	2) Determine wave amplitude from baseline
	3) Record amplitude/deviation for future analysis
...
