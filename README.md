# SCALE
A public source code for SCALE: Toward Multiple Specialty Learners for Explaining GNNs via Online Knowledge Distribution (Please check the pdf file for the preprint version)

<!-- This is a part of another project. We have just copied the folder to a new branch for the submission purpose. Therefore, -->
We are cleaning and restructure our code. 

You can refer to notebook files for visualization and results. 

Please refer to README files in ./node and ./graph for training and explanation. Thank you very much ^^!

We have implemented an interactive web application based on algorithms proposed in this paper and named it INGREX. Please checkout [https://github.com/alexbui91/INGREX](https://github.com/alexbui91/INGREX).

[![Video Demo](https://res.cloudinary.com/marcomontalbano/image/upload/v1666665645/video_to_markdown/images/youtube--3T2TojvBs0w-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/3T2TojvBs0w "")


## Important Notice
Lately, Amara et al., 2022 (https://arxiv.org/abs/2206.09677) have listed up some explanation methods for node classification including PageRank (A version of Random Walk with Restart). Their first pre-print version was submitted to ArXiv in June, 2022. However, we have a proof that our proposal for using RWR for explaining node classification was first introduced in May 2022 (https://github.com/alexbui91/pgx). We only found out about their existence recently after the submission of our paper to ICDE 2023. ***Therefore, we do believe that we still get credit for this contribution.***

## Must-installed libraries:
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


