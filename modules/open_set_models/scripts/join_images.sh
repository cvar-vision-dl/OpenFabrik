#!/bin/bash

# Script to join images with matching names from two folders
# Usage: ./join_images.sh <folder1> <folder2> [output_folder] [direction]
# direction: horizontal (default) or vertical

if [ $# -lt 2 ]; then
    echo "Usage: $0 <folder1> <folder2> [output_folder] [direction]"
    echo "direction: horizontal (default) or vertical"
    exit 1
fi

folder1="$1"
folder2="$2"
output_folder="${3:-.}"
direction="${4:-horizontal}"

if [ ! -d "$folder1" ]; then
    echo "Error: folder1 '$folder1' does not exist"
    exit 1
fi

if [ ! -d "$folder2" ]; then
    echo "Error: folder2 '$folder2' does not exist"
    exit 1
fi

mkdir -p "$output_folder"

# Image extensions to search for
extensions="jpg jpeg png gif bmp webp"

count=0

# Find all images in folder1
for file1 in "$folder1"/*; do
    if [ -f "$file1" ]; then
        filename=$(basename "$file1")
        name_only="${filename%.*}"
        ext="${filename##*.}"

        # Skip if not an image extension
        if ! echo "$extensions" | grep -q "$ext"; then
            continue
        fi

        # Look for matching file in folder2
        file2=""
        for e in $extensions; do
            if [ -f "$folder2/$name_only.$e" ]; then
                file2="$folder2/$name_only.$e"
                break
            fi
        done

        if [ -n "$file2" ]; then
            output_file="$output_folder/${name_only}_joined.png"

            if [ "$direction" = "vertical" ]; then
                convert "$file1" "$file2" -append "$output_file"
            else
                convert "$file1" "$file2" +append "$output_file"
            fi

            if [ $? -eq 0 ]; then
                echo "✓ Joined: $filename"
                ((count++))
            else
                echo "✗ Failed: $filename"
            fi
        fi
    fi
done

echo "Completed: $count images joined"