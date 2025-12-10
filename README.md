# spark-research
This project revolved aroudn training a machine vision model on welding videos to recognizes sparks, with the goal of automatically turning off failing power transformers. The original Pytorch has been converted to tflite, quantized, and optimized with vela for the VAR SOM MX93 board. The model's architecture is based on MobileNet.
<video width="640" height="480" controls>
  <source src="spark-detector-demo.mp4" type="video/mp4">
Your browser does not support the video tag.
</video> 
The above video shows the model being run on a test video. The -128 output indicates high confidence in the presence of sparks. When the man comes onto the screen, the positive numbers indicate confidence in the absence of sparks.
