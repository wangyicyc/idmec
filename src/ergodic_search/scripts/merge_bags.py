#!/usr/bin/env python3
import rosbag
import sys
import os

if len(sys.argv) < 3:
    print("Usage: python merge_bags.py output.bag input1.bag input2.bag ...")
    sys.exit(1)

output_bag = sys.argv[1]
input_bags = sys.argv[2:]

with rosbag.Bag(output_bag, 'w') as outbag:
    for input_bag in input_bags:
        if not os.path.exists(input_bag):
            print(f"Warning: {input_bag} does not exist. Skipping.")
            continue
        print(f"Merging {input_bag}...")
        for topic, msg, t in rosbag.Bag(input_bag).read_messages():
            outbag.write(topic, msg, t)

print(f"Successfully merged into {output_bag}")