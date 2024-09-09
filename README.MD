# fastpose

This is the remake of the main repository. This provides a way to make your custom dataset using Fastpose and get keypoints and the respective labels in separate files.

Steps:
1. Put all your video files under the data folder under respective labels. For example, all videos related to label 1 should be in "1" folder and videos for label 2 should be in folder "2" under data.

2. Usage: python demo_2d.py <directory_path> [max_persons]
Example: Usage: python demo_2d.py /path/to/class/1 [max_persons]

3. Replace 1 in step 2 with all classes one by one.
You will have all the keypoints and respective labels stored in /keypoints folder

We initiated fastpose to be an open source library that can perform a 2D/3D pose estimation in real time on a CPU.


Read the full project description at :

[https://drnoodle.github.io/fastpose_html/](https://drnoodle.github.io/fastpose_html/)


download the parameters at :

[https://bitbucket.org/hugo_bonnome/fastpose_parameters/downloads/parameters.zip](https://bitbucket.org/hugo_bonnome/fastpose_parameters/downloads/parameters.zip)

