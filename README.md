isoP
================

## Introduction
isoP was orignally developed in December 2014 by C. Delavau as a way of creating gridded precipitation isotopic data. It was developed in matlab which is unfortunately not open source. This is a python version of the original program converted by J. Gray in 2023.
## Installation
In order to run isoP you need to make sure you have downloaded python 3.10 or higher. You can download python from [here](https://www.python.org/downloads/). Once you have downloaded python you need to install the following packages:
* numpy
* pandas
* netCDF4
* scipy

You can do this by loading up the command prompt and typing the following for each package:
```
pip install [package name]
```
Afterwards you will need to clone this repository to your system. You can place the folder where you like, but I suggest placing it in the same folder as the main basin directory. This is because you will need to enter the path to the main basin directory when you run the program.

## How to use
Once the correct packages have been installed as well as python, you have two options for running it:
1. Your favourite IDE
2. Command prompt

### IDE
Open the folder containg the isoP file inside the IDE, from the command prompt type:
```
isoP.py
```
### Command prompt
Open the command prompt and navigate to the folder containing the isoP file. Type the following:
```
python isoP.py
```
After you have done either of these you will be prompted to enter the following:
* The path to the main basin directory
* The start year
* The end year

Keep an eye out for more prompts that may appear later.

# WARNING
This program is still in the beginning stages of development. While it does work it is worth noting that it is not capable of working with UTM coordinates as of yet, so please ensure that you are using latitude and longitude coordinates.

Also, while the python version does currently run with the linear regression models that were used in the orignal matlab version, it lacks the same functionality as those models. It loads in CSV files that contain the linear regression coefficients and intercepts, this will be remedied in the future with a better solution.

It is missing the last function, as I am currently working on a way to implement it in python. I will update this when I have a solution.


