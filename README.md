# Automatic positioning for sewer defects 
### computing geographic coordinate of sewer defects
<p><a target="_blank" href="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" style="display: inline-block;"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="42" height="42" /></a>
<a target="_blank" href="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" style="display: inline-block;"><img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="42" height="42" /></a>
<a target="_blank" href="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" style="display: inline-block;"><img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="42" height="42" /></a></p>

Directories and files include：
* The directory named 'Dyhead training' contains files related to training object detection models.
  * The file named _atss_swin-l-p4-w12_fpn_dyhead_ms-2x_sewer.py is training config file of dyhead under openmmlab platform.
  * The file named 20240524_230043.log is the training log file for object detection model(dyhead).
  * The trained weight file of object detection:[Dyhead.pth](https://drive.google.com/file/d/1rggV4CXF4t9gJV0VaR2xz-pH8IR8rrlN/view?usp=sharing "Dyhead.pth"). This file should be placed in the checkpoints directory under this project.
* The directory named 'experimental statistical data' contains files related to the experimental results of the paper.
  * The fle named sewer.db is the database for statisticing the results of object tracking metrics. It is a SQLite format database file.
  * The file named coordinate.pkl is a binary format file that stores the coordinate values of all defects calculated for an example video. The generation of this file can refer to the code in coordinate_computing. py. If you need to view the content of the file, you can refer to the data structure before serialization in the program code to deserialize it.
* The file named requirements.txt describes the packages required for the conda virtual environment
* The file named dinov2-main.zip is a compressed file of the source code for depth estimation. Because the openmmlab library used for depth estimation is an earlier version and is not compatible with the latest version of openmmlab library used the sewer project, it is separated as another python project. The sewer project will perform overall process control and call the code in the dinov2-main project for depth estimation. Before being able to run the code, the weight files required for depth estimation should be downloaded and placed in the models directory of the dinov2-main project.

