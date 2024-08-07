# Automatic positioning for sewer defects 
### computing geographic coordinate of sewer defects


* The file named _atss_swin-l-p4-w12_fpn_dyhead_ms-2x_sewer.py is training config file of dyhead under openmmlab platform.
* The file named 20240524_230043.log is the training log file for object detection model(dyhead).
* The files named mmapps4sewer.zip and dinov2-main.zip are the source code. Because the openmmlab library used in dinov2 is an earlier version and is not compatible with the latest version of openmmlab library used by mmapps4sewer, it is separated as a separate project. The mmapps4Sewer will perform overall process control and call the code in the dinov2-main project for depth estimation. Before being able to run the code, the weight files required for depth estimation should be downloaded and placed in the models directory of the dinov2-main project.
* The fle named sewer.db is the database for statisticing the results of object tracking metrics.
* The trained weight file of object detection:[Dyhead.pth](https://drive.google.com/file/d/1rggV4CXF4t9gJV0VaR2xz-pH8IR8rrlN/view?usp=sharing "Dyhead.pth"). This file should be placed in the checkpoints directory under the mmapps4sewer project.
