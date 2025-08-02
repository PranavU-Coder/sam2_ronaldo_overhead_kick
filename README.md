# sam2_ronaldo_overhead_kick
using SAM2 from meta to annotate parts of initial frame and get entire object annotated throughout the video

<p align="center">
  <img src="model_diagram.png" alt="How the model works" width="400">
  <br>
  <em>How the model works (refer to the notebooks on official page)</em>
</p>

to annotate any video , first split the video into frames using ffmpeg

to split :

ffmpeg -i ENTER_VIDEO_NAME.mp4 -q:v 2 -start_number 0 %05d.jpg

once frames are annotated , go to the directory where annotated frames are getting saved and run :

ffmpeg -framerate 30 -i s%d.png -c:v libx264 -r 30 output.mp4
