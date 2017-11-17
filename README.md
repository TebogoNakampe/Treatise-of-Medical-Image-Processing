# Treatise of Medical Image Processing (TMIP)
# Volume 1: Brain Tumour Detection

<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/Treatise-of-Medical-Image-Processing/blob/master/Brain.jpg">
</p>

In this project, I built a model to classify brain tumours into three types based on MRI scans: Astrocytoma, Oligodendroglioma or Glioblastoma.

From a dataset of 32 patients, tumour features such as size, enhancement quality, necrosis proportion, etc. were extracted by radiologists. Diagnosis was also established for these patients. Based on this information I was able to create an optimised model to classify tumours with a 90% cross-validated accuracy.

# Data Preprocessing and Model Selection:

                                          TMIP.ipynb
                                          Data = REMBRANDT may be found at (https://wiki.cancerimagingarchive.net/display/Public/REMBRANDT#4b0fe4760f6d405e9d09ad75c6f54790)
                                                                                                         
<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/Treatise-of-Medical-Image-Processing/blob/master/output_75_1.png">
</p>
The main approach was to extract tumour features from 120 patients MRI scans for whom diagnosis was established by neurologis. A detailed notebook containing an exploration of several image processing methods can be found in TMIP.ipynb

TMIP using FCN and Simple CNN:
In order apply this model to new patients and generate an unbiased estimate of the model's performance, we are exploring simple convolutional neural networks and Fully Convolutional Neural Networks. However while we did manage to extract features such as tumour dimensions, side of epicentre, T1/FLAIR ratio and Enhancement Quality, so far we have been unable to extract features such as necrosis proportion or thickness of enhancing margin. 

Data Preprocessing then FCN and simple CNN:
                simple_cnn.ipynb
                fcn.ipynb
                


