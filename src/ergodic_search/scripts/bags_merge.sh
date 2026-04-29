#!/bin/bash

# python pc.py
sleep 2
python merge_bags.py global.bag path0.bag path1.bag path2.bag path3.bag traj0.bag traj1.bag traj2.bag traj3.bag map.bag
sleep 2

python merge_bags.py robot_0.bag path0-0.bag path1-0.bag path2-0.bag path3-0.bag traj0-0.bag traj1-0.bag traj2-0.bag traj3-0.bag map_r0.bag
sleep 2
python merge_bags.py robot_1.bag path0-1.bag path1-1.bag path2-1.bag path3-1.bag traj0-1.bag traj1-1.bag traj2-1.bag traj3-1.bag map_r1.bag
sleep 2
python merge_bags.py robot_2.bag path0-2.bag path1-2.bag path2-2.bag path3-2.bag traj0-2.bag traj1-2.bag traj2-2.bag traj3-2.bag map_r2.bag
sleep 2
python merge_bags.py robot_3.bag path0-3.bag path1-3.bag path2-3.bag path3-3.bag traj0-3.bag traj1-3.bag traj2-3.bag traj3-3.bag map_r3.bag