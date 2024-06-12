#!/bin/bash
if [ "$1" != "" ]; then
	dataset_dir=`cd "$1"; pwd`;
else
    echo "Usage: ./download_ijrr_sugar_beet_2016_rosbag_data.sh path_to_destination_folder"
    exit 1
fi
metadata_dir=`pwd`/metadata
list_days_file=$metadata_dir/list_days.txt
server="http://www.ipb.uni-bonn.de/datasets_IJRR2017/rosbags"
rosbags_data_root=$dataset_dir/rosbags
mkdir $rosbags_data_root
cd $rosbags_data_root
while read -r day || [ -n "$day" ]
do
    day_chunks_list_file=$metadata_dir/$day.txt
    day_folder=$rosbags_data_root/$day
 	mkdir $day_folder
 	cd $day_folder
	while read -r chunk || [ -n "$chunk" ]
	do
		echo "Downloading $chunk from server."
		chunk_link=$server/$day/$chunk.bag
		echo chunk_link
		wget $chunk_link
	done < "$day_chunks_list_file"
done < "$list_days_file"
