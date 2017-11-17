# Treatise of Medical Image Processing (TMIP)
# Volume 1: Brain Tumour Detection

<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/Treatise-of-Medical-Image-Processing/blob/master/Brain.jpg">
</p>

In this project, I built a model to classify brain tumours into three types based on MRI scans: Astrocytoma, Oligodendroglioma or Glioblastoma.

From a dataset of 32 patients, tumour features such as size, enhancement quality, necrosis proportion, etc. were extracted by radiologists. Diagnosis was also established for these patients. Based on this information I was able to create an optimised model to classify tumours with a 90% cross-validated accuracy.https://github.com/TebogoNakampe/Treatise-of-Medical-Image-Processing/blob/master/output_75_1.png

# Data Preprocessing and Model Selection:

                                          TMIP.ipynb
                                          Data = REMBRANDT CIA
                                                                                                         (https://wiki.cancerimagingarchive.net/display/Public/REMBRANDT#4b0fe4760f6d405e9d09ad75c6f54790)
<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/Treatise-of-Medical-Image-Processing/blob/master/output_75_1.png">
</p>
The main approach was to extract tumour features from 120 patients MRI scans for whom diagnosis was established by neurologis. A detailed notebook containing an exploration of several image processing methods can be found in TMIP.ipynb

I intended to apply my model to these new patients and generate an unbiased estimate of my model's performance. However while I did manage to extract features such as tumour dimensions, side of epicentre, T1/FLAIR ratio and Enhancement Quality, so far I have been unable to extract features such as necrosis proportion or thickness of enhancing margin. I will keep working on this project regularly to improve my feature extraction techniques.

I am also planning to use a CNN as an alternative classifying method.


References
1. Scarpace, Lisa, Flanders, Adam E., Jain, Rajan, Mikkelsen, Tom, & Andrews, David W. (2015). Data From REMBRANDT. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2015.588OZUZB
2. Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057
