#!/bin/bash
while true; do
    python3 ./trt_video_demo.py --video ms03_vid.mp4
    python3 ./trt_video_demo.py --video kof00.mp4
    python3 ./trt_video_demo.py --video sp.mp4
done