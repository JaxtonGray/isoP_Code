isoP
================

## Introduction
isoP was orignally developed in 2014 by ***insert names here*** as a way of creating gridded files for isotopes .... *Add more Jax*

## Installation
In order to run isoP you need to make sure you have downloaded python 3.6 or higher. You can download python from [here](https://www.python.org/downloads/). Once you have downloaded python you need to install the following packages:
* numpy
* pandas
* netCDF4

You can do this by loading up the command prompt and typing the following:
```
pip install [package name]
```
Once you have installed these packages you can download you can run isoP in your favoured IDE or your command prompt by typing:
```
python isoP.py
```
***This is not fully correct Jax***

## Note
This program is still in the beginning stages of development. While it does work it is worth noting that it is not capable of working with UTM coordinates as of yet, so please ensure that you are using latitude and longitude coordinates.

Also, while the python version does currently run with the linear regression models that were used in the orignal matlab version, it lacks the same functionality as those models. It loads in CSV files that contain the linear regression coefficients and intercepts, this will be remedied in the future with a better solution.

It is missing the last function, as I am currently working on a way to implement it in python. I will update this when I have a solution.


