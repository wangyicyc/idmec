#!/bin/bash
# python attachment.py 
# sleep 2
# python pc.py
# sleep 2

# bash bags_merge.sh
# sleep 2

# mv *.bag ../datas/bags
python render_bag_to_video_global.py
sleep 2
python render_bag_to_video.py 0
sleep 2
python render_bag_to_video.py 1
sleep 2
python render_bag_to_video.py 2
sleep 2
python render_bag_to_video.py 3