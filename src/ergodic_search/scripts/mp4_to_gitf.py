import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

from moviepy.video.io.VideoFileClip import VideoFileClip
# ... rest of your codeip

# 加载视频
clip = VideoFileClip("global.mp4")
clip.write_gif("global.gif", fps=10)
clip = VideoFileClip("robot0.mp4")
clip.write_gif("robot0.gif", fps=10)
clip = VideoFileClip("robot1.mp4")
clip.write_gif("robot1.gif", fps=10)
clip = VideoFileClip("robot2.mp4")
clip.write_gif("robot2.gif", fps=10)
clip = VideoFileClip("robot3.mp4")
clip.write_gif("robot3.gif", fps=10)