#!/bin/bash
# 循环10次
for count in {1..2}
do
    # 构建上级目录中的文件夹路径
    folder_name="../case${count}"
    # 创建文件夹（使用-p参数确保上级目录存在）
    mkdir -p "$folder_name"
    # mkdir "../figures"
    # touch "../config/random_map_history.jsonl"
    echo "已创建文件夹: $folder_name"
    
    # 执行Python脚本
    python3 baseline1.py
    sleep 3
    python3 baseline2.py
    sleep 3
    python3 pc.py
    mv ../datas/config/random_map_history.jsonl "$folder_name/"
    mv ../figures/* "$folder_name/."
    mv ergodic_metric.xlsx "$folder_name/"
    mv ../datas/logs/app.log "$folder_name/"
    # python merge_bags.py global.bag path0.bag path1.bag path2.bag path3.bag traj0.bag traj1.bag traj2.bag traj3.bag map.bag
    # mv global.bag "$folder_name/"
    # rm path0.bag path1.bag path2.bag path3.bag traj0.bag traj1.bag traj2.bag traj3.bag map.bag
    # 重新创建random_map_history.jsonl文件以供下一次使用
    touch "../datas/logs/app.log"
    touch "../datas/config/random_map_history.jsonl"
    echo "------------------------"
done
python plot_result.py
echo "所有任务完成！案例文件夹已创建在上级目录中。"