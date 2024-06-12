#!/bin/bash

if [ "$1" != "" ]; then
    # Use readlink to get the absolute path
    dataset_dir=$(readlink -f "$1")
    if [ $? -ne 0 ]; then
        echo "Invalid path: $1"
        exit 1
    fi

     # Check if the directory exists
    if [ ! -d "$dataset_dir" ]; then
        echo "Directory $dataset_dir does not exist. Creating it..."
        mkdir -p "$dataset_dir"
        
        if [ $? -eq 0 ]; then
            echo "Directory $dataset_dir created successfully."
        else
            echo "Failed to create directory $dataset_dir."
            exit 1
        fi
    else
        echo "Directory $dataset_dir already exists."
    fi
else
    echo "Usage: ./download_ijrr_sugar_beet_2016_raw_data.sh path_to_destination_folder"
    exit 1
fi
echo "Downloading to $dataset_dir"
metadata_dir=`pwd`/metadata
list_days_file=$metadata_dir/list_days.txt
server="http://www.ipb.uni-bonn.de/datasets_IJRR2017/raw_data"
raw_data_root=$dataset_dir/raw_data

# Check if the directory exists
if [ ! -d "$raw_data_root" ]; then
    echo "Directory $raw_data_root does not exist. Creating it..."
    mkdir -p "$raw_data_root"
    
    if [ $? -eq 0 ]; then
        echo "Directory $raw_data_root created successfully."
    else
        echo "Failed to create directory $raw_data_root."
        exit 1
    fi
else
    echo "Directory $raw_data_root already exists."
fi

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
