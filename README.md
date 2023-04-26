# Optical Mark Recognition (OMR) MCQ Automated Grading

## Technologies Used

<table><tr><td valign="top" width="33%">
<div align="center">  
<a href="https://opencv.org/" target="_blank"><img style="margin: 10px" src="https://profilinator.rishav.dev/skills-assets/opencv-icon.svg" alt="OpenCV" height="50" /></a>  
<a href="https://www.python.org/" target="_blank"><img style="margin: 10px" src="https://profilinator.rishav.dev/skills-assets/python-original.svg" alt="Python" height="50" /></a>  </div>
</td></tr></table>

## Work Pipeline

The program works as follows:

- Reads the original image in grayscale.
- Finds the edges.
- Finds the contours, from which we extract the biggest rectangle's cornerpoints.
- The image is processed using warp to get the bird's eye view.
- Perform a threshold
- Find the marks
- Saves the marks in a `.csv` file

