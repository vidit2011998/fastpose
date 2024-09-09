# FastPose

## Overview

FastPose is an open-source library for real-time 2D/3D pose estimation on CPU. This repository is a remake of the main project, providing a method to create custom datasets using FastPose and generate keypoints with their respective labels in separate files.

## Steps to Use FastPose

1. **Data Organization**:
   - Put all your video files under the `data` folder, grouped by labels. For example:
     ```
     data/
     ├── 1/
     │   ├── video1.mp4
     │   ├── video2.mp4
     │   └── ...
     ├── 2/
     │   ├── video1.mp4
     │   ├── video2.mp4
     │   └── ...
     └── ...
     ```
  - Here 1,2 .... are labels

2. **Running the Script**:
   - Use the following command to process videos:
     ```bash
     python demo_2d.py <directory_path> [max_persons]
     ```
   - Example:
     ```bash
     python demo_2d.py /path/to/class/1 [max_persons]
     ```

3. **Processing All Classes**:
   - Repeat step 2 for each class folder, replacing the directory path accordingly.

## Output

After processing, you will find all keypoints and their respective labels stored in the `/keypoints` folder.

## Additional Resources

- **Full Project Description**: [https://drnoodle.github.io/fastpose_html/](https://drnoodle.github.io/fastpose_html/)
- **Download Parameters**: [https://bitbucket.org/hugo_bonnome/fastpose_parameters/downloads/parameters.zip](https://bitbucket.org/hugo_bonnome/fastpose_parameters/downloads/parameters.zip)

FastPose is designed to be an open-source library that can perform 2D/3D pose estimation in real-time on a CPU.
