# Machine Vision Research on Spark Recognition
This project revolved aroudn training a machine vision model on welding videos to recognizes sparks, with the goal of automatically turning off failing power transformers. The original Pytorch has been converted to tflite, quantized, and optimized with vela for the VAR SOM MX93 board. The model's architecture is based on MobileNet.

The model outputs numbers between -127 and 128, with -127 being the highest confidence of the presence of sparks, and 128 being the highest confidence of their absence.
