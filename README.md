# sewer
computing geographic coordinate of sewer defects


1、The file named _atss_swin-l-p4-w12_fpn_dyhead_ms-2x_sewer.py is training config file of Dyhead under Openmmlab platform.
2、The file named 20240524_230043.log is the training log file for object detection.
3、The files named mmapps4sewer2024-07-17.zip and dinov2-main2024-07-17.zip are the source code.Because the openmmlab library used in Dinov2 is an earlier version of mmcv-full1.5.0 and is not compatible with the latest version of openmmlab library used by mmapps4sewer, it is separated as a separate project.The mmapps4Sewer will perform overall process control and call the code in the dinov2-main project for depth estimation.Before being able to run the code, the weight files required for depth estimation should be downloaded and placed in the models directory of the dinov2-main project
4. The file named Dyhead.pth is the trained weight file of object detection. This file should be placed in the checkpoints directory under the mmapps4sewer project.
5、The fle named sewer.db is the database for statisticing the results of object tracking metrics.
