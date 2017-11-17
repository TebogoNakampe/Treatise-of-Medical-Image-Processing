# Treatise of Medical Image Processing (TMIP)



In this project, I built a model to classify brain tumours into three types based on MRI scans: Astrocytoma, Oligodendroglioma or Glioblastoma.

From a dataset of 32 patients, tumour features such as size, enhancement quality, necrosis proportion, etc. were extracted by radiologists. Diagnosis was also established for these patients. Based on this information I was able to create an optimised model to classify tumours with a 90% cross-validated accuracy.

The exploratory data analysis and model selection can be found in the [main notebook](Project - EDA, Model selection, Image processing.ipynb). Data was taken from the REMBRANDT study on the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/REMBRANDT#4b0fe4760f6d405e9d09ad75c6f54790)[1,2].

I then tried to extract tumour features from 95 additional patients MRI scans for whom diagnosis was established by radiologists but tumour features were not extracted. A detailed notebook containing an exploration of several image processing methods can be found in the [Image Processing research](Image processing research/Capstone - Image processing workflow.ipynb) folder. A summary of these techniques has been included in the [main notebook](Project - EDA, Model selection, Image processing.ipynb).

I intended to apply my model to these new patients and generate an unbiased estimate of my model's performance. However while I did manage to extract features such as tumour dimensions, side of epicentre, T1/FLAIR ratio and Enhancement Quality, so far I have been unable to extract features such as necrosis proportion or thickness of enhancing margin. I will keep working on this project regularly to improve my feature extraction techniques.

I am also planning to use a CNN as an alternative classifying method.


References
1. Scarpace, Lisa, Flanders, Adam E., Jain, Rajan, Mikkelsen, Tom, & Andrews, David W. (2015). Data From REMBRANDT. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2015.588OZUZB
2. Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057
