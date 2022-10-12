# SCALE
A public source code for SCALE

This is a part of another project. We have just copied the folder to a new branch for the submission purpose. Therefore, we are cleaning and restructure our code.

You can refer to notebook files for visualization and results. 

We will update guidelines for running ASAP. Thank you very much ^^!


Must-installed libraries:
- Pytorch
- DGL
- Numpy
- Matplotlib
- SKlearn
- SHAP (for DeepLIFT Pytorch)

Run the following line to include the source code since our code uses hard directory:
```
pip install --user -e ./
```

Source code & how to train a blackbox with explainers:
We separate source codes of node classification and graph classification in node/ and graph/ folders. In each folder, there is a README file to describe python files.

How to explain:
Please refer to notebooks folders in node/ and graph/

File descriptions:
- shared_networks.py: some networks shared among graph & node classification
- utils: some utilities shared among graph & node classification

