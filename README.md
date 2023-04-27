# Optical Mark Recognition (OMR) MCQ Automated Grading

## Technologies Used

<div align="center">
<a href="https://opencv.org/" target="_blank"><img style="margin: 10px" src="https://profilinator.rishav.dev/skills-assets/opencv-icon.svg" alt="OpenCV" height="50" /></a>  
<a href="https://www.python.org/" target="_blank"><img style="margin: 10px" src="https://profilinator.rishav.dev/skills-assets/python-original.svg" alt="Python" height="50" /></a>
</div>

## Work Pipeline

*For Educational Purposes ONLY*

The code works as follows:

- Reads the original image in grayscale.
- Finds the edges.
- Finds the contours, from which we extract the biggest rectangle's cornerpoints.
- The image is processed using warp to get the bird's eye view.
- Perform a threshold
- Find the shaded/marked boxes/bubbles
- Generates an answer key from a solution image
  - This is an option step, as we can define any answer key in a specific format.
- Saves the grades in a `.csv` file

___

## Install

Clone the repository, or alternatively download it

```shell
git clone https://github.com/Bakhomious/OMR-MCQ-Automated-Grading
cd OMR-MCQ-Automated-Grading
```

Install pip requirements

```shell
python -m pip install --user -r requirements.txt
```

Run
```shell
python main.py
```