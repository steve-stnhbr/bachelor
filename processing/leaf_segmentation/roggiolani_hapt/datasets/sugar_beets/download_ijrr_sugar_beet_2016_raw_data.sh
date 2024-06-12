#!/bin/bash
if [ "$1" != "" ]; then
	dataset_dir=`cd "$1"; pwd`;
else
    echo "Usage: ./download_ijrr_sugar_beet_2016_raw_data.sh path_to_destination_folder"
    exit 1
fi
metadata_dir=`pwd`/metadata
list_days_file=$metadata_dir/list_days.txt
server="http://www.ipb.uni-bonn.de/datasets_IJRR2017/raw_data"
raw_data_root=$dataset_dir/raw_data
mkdir $raw_data_root
cd $raw_data_root
while read -r day || [ -n "$day" ]
do
    day_chunks_list_file=$metadata_dir/$day.txt
    day_folder=$raw_data_root/$day
 	mkdir $day_folder
 	cd $day_folder
	while read -r chunk || [ -n "$chunk" ]
	do
		echo "Downloading $chunk from server."
		chunk_link=$server/$day/$chunk.zip
		echo chunk_link
		wget $chunk_link
	done < "$day_chunks_list_file"
done < "$list_days_file"
