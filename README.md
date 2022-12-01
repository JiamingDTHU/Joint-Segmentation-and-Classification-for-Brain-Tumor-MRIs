# Joint-Segmentation-and-Classification-for-Brain-Tumor-MRIs
This project is the midterm project of course fMRI Principles and Analytics (2022-autumn) at Tsinghua SIGS. 

---

## Dataset Description

Here is the descriptive information about the dataset from its builder. 

>This brain tumor dataset containing 3064 T1-weighted contrast-inhanced imagesfrom 233 patients with three kinds of brain tumor: meningioma (708 slices), glioma (1426 slices), and pituitary tumor (930 slices). Due to the file sizelimit of repository, we split the whole dataset into 4 subsets, and achive them in 4 .zip files with each .zip file containing 766 slices.The 5-fold cross-validation indices are also provided.

>This data is organized in matlab data format (.mat file). Each file stores a struct containing the following fields for an image:
>>**cjdata.label:** 1 for meningioma, 2 for glioma, 3 for pituitary tumor
>>**cjdata.PID:** patient ID
>>**cjdata.image:** image data
>>**cjdata.tumorBorder:** a vector storing the coordinates of discrete points on tumor border.
>>*For example, [x1, y1, x2, y2,...] in which x1, y1 are planar coordinates on tumor border.*
>>*It was generated by manually delineating the tumor border. So we can use it to generate binary image of tumor mask.*
>>**cjdata.tumorMask:** a binary image with 1s indicating tumor region

>Jun Cheng
School of Biomedical Engineering
Southern Medical University, Guangzhou, China
[GitHub repository](https://github.com/chengjun583/brainTumorRetrieval)
Email: chengjun583@qq.com

---

## Plan

 1. We intend to reproduce the experiment results of a paper using cUNet that can simultaneously complete segmentation task and classification task[^01]. 
 2. After we complete task 1, we will modify our networks to try to achieve higher prediction performance 

## Todo List
- [x] Data loading module
- [x] Network definition
- [x] Loss function definition
- [x] Training and evaluation function definition
- [x] Main process (training and evaluation)
- [ ] Debug and optimization


[^01]:Simultaneous Segmentation and Classification of Bone Surfaces from Ultrasound Using a Multi-feature Guided CNN. Wang, P et al