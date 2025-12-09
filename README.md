# spark-research
Machine vision model trained on welding videos to recognizes sparks with the goal of automatically turning off failing power transformers. Optimized for the VAR SOM MX93 board.i
 
<video width="320" height="240" controls>
  <source src="spark-detector-demo.mp4" type="video/mp4">
Your browser does not support the video tag.
</video> 

The above video shows the model being run on a test video. The -128 output indicates high confidence in the presence of sparks. When the man comes onto the screen, the positive numbers indicate confidence in the absence of sparks.
