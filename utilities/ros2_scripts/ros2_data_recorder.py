#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, CompressedImage
from cv_bridge import CvBridge
import cv2
import json
import os
import time
import argparse
from datetime import datetime
import message_filters
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
from tqdm import tqdm
import threading
import sys
import io
from PIL import Image as PILImage
import numpy as np
import imageio


class DataRecorder(Node):
    def __init__(self, output_dir, skip_frames, duration,
                 camera_info_topic, image_topic, laserscan_topic, pointcloud_topic,
                 image_type='compressed',
                 record_video=True, record_frames=True,
                 record_camera_info=True, record_laserscan=True, record_pointcloud=True,
                 show_progress=True):
        super().__init__('data_recorder')

        # Parameters
        self.output_dir = output_dir
        self.skip_frames = skip_frames if skip_frames is not None else 0
        self.duration = duration
        self.camera_info_topic = camera_info_topic
        self.image_topic = image_topic
        self.laserscan_topic = laserscan_topic
        self.pointcloud_topic = pointcloud_topic
        self.image_type = image_type.lower()
        self.record_video = record_video
        self.record_frames = record_frames
        self.record_camera_info = record_camera_info
        self.record_laserscan = record_laserscan
        self.record_pointcloud = record_pointcloud
        self.show_progress = show_progress

        # Create output directories
        self.create_directories()

        # Initialize variables
        self.bridge = CvBridge()
        self.camera_info_received = False
        self.start_time = None
        self.video_start_time = None
        self.frame_count = 0  # Total frames received
        self.saved_frame_count = 0  # Frames actually saved
        self.skip_counter = 0  # Counter for frame skipping logic
        self.video_frames = []
        self.frame_timestamps = []
        self.is_shutdown = False

        # Add buffers for synchronized data
        self.latest_laserscan = None
        self.latest_pointcloud = None
        self.laserscan_frame_stamp = None
        self.pointcloud_frame_stamp = None

        # Initialize progress bar if duration is set and progress is enabled
        self.progress_bar = None
        self.progress_thread = None
        if self.duration is not None and self.show_progress:
            self.setup_progress_bar()

        # Timer for managing recording duration (if specified)
        if self.duration is not None:
            self.timer = self.create_timer(1.0, self.timer_callback)  # Check every second

        # Setup subscriptions
        if self.record_camera_info and self.camera_info_topic:
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                10)

        if (self.record_frames or self.record_video) and self.image_topic:
            # Subscribe based on image type
            if self.image_type == 'compressed':
                self.image_sub = self.create_subscription(
                    CompressedImage,
                    self.image_topic,
                    self.image_callback,
                    10)
                self.get_logger().info(f'Subscribing to compressed image topic: {self.image_topic}')
            elif self.image_type == 'raw':
                self.image_sub = self.create_subscription(
                    Image,
                    self.image_topic,
                    self.image_callback,
                    10)
                self.get_logger().info(f'Subscribing to raw image topic: {self.image_topic}')
            else:
                self.get_logger().error(f'Invalid image type: {self.image_type}. Must be "compressed" or "raw"')
                raise ValueError(f'Invalid image type: {self.image_type}')

        if self.record_laserscan and self.laserscan_topic:
            self.laserscan_sub = self.create_subscription(
                LaserScan,
                self.laserscan_topic,
                self.laserscan_buffer_callback,
                10)

        if self.record_pointcloud and self.pointcloud_topic:
            self.pointcloud_sub = self.create_subscription(
                PointCloud2,
                self.pointcloud_topic,
                self.pointcloud_buffer_callback,
                10)

        # Log
        skip_desc = f"every {self.skip_frames + 1}" if self.skip_frames > 0 else "all"
        self.get_logger().info(f'Data recorder initialized - saving {skip_desc} frames, ' +
                               f'Duration: {duration if duration else "unlimited"}s')
        self.get_logger().info(f'Output directory: {output_dir}')
        self.get_logger().info(f'Image type: {self.image_type}')
        self.get_logger().info(f'Topics: Camera info: {camera_info_topic}, Image: {image_topic}, ' +
                               f'LaserScan: {laserscan_topic}, PointCloud: {pointcloud_topic}')

    def create_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_dir, exist_ok=True)

        if self.record_frames:
            self.frames_dir = os.path.join(self.output_dir, 'frames')
            os.makedirs(self.frames_dir, exist_ok=True)

        if self.record_laserscan:
            self.laserscan_dir = os.path.join(self.output_dir, 'laserscan')
            os.makedirs(self.laserscan_dir, exist_ok=True)

        if self.record_pointcloud:
            self.pointcloud_dir = os.path.join(self.output_dir, 'pointcloud')
            os.makedirs(self.pointcloud_dir, exist_ok=True)

    def should_save_frame(self):
        """Determine if the current frame should be saved based on skip_frames setting"""
        # Simple frame skipping logic:
        # skip_counter counts up from 0
        # When it reaches skip_frames, we save the frame and reset counter

        if self.skip_counter >= self.skip_frames:
            self.skip_counter = 0  # Reset counter
            return True
        else:
            self.skip_counter += 1  # Increment skip counter
            return False

    def setup_progress_bar(self):
        """Setup progress bar in a separate thread"""

        def progress_updater():
            # Create progress bar with duration in seconds
            self.progress_bar = tqdm(total=int(self.duration),
                                     desc="Recording",
                                     unit="s",
                                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} seconds",
                                     file=sys.stdout)

            # Update progress bar until recording is complete
            while not self.progress_bar.n >= self.progress_bar.total and not self.is_shutdown:
                if self.start_time is not None:
                    current_time = self.get_clock().now().seconds_nanoseconds()[0]
                    elapsed_time = current_time - self.start_time
                    self.progress_bar.n = min(int(elapsed_time), self.progress_bar.total)
                    self.progress_bar.refresh()
                time.sleep(0.1)

            # Ensure progress bar reaches 100%
            if not self.is_shutdown and self.progress_bar is not None:
                self.progress_bar.n = self.progress_bar.total
                self.progress_bar.refresh()

        # Start progress bar in a separate thread
        self.progress_thread = threading.Thread(target=progress_updater)
        self.progress_thread.daemon = True
        self.progress_thread.start()

    def check_duration(self):
        """Check if recording duration has elapsed"""
        if self.duration is None:
            # Unlimited duration
            return False

        if self.start_time is None:
            # Recording hasn't started yet
            return False

        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed_time = current_time - self.start_time

        return elapsed_time >= self.duration

    def camera_info_callback(self, msg):
        """Process camera info message and save as JSON"""
        if not self.camera_info_received:
            camera_info = {
                'width': msg.width,
                'height': msg.height,
                'distortion_model': msg.distortion_model,
                'D': list(msg.d),
                'K': list(msg.k),
                'R': list(msg.r),
                'P': list(msg.p),
                'binning_x': msg.binning_x,
                'binning_y': msg.binning_y,
                'roi': {
                    'x_offset': msg.roi.x_offset,
                    'y_offset': msg.roi.y_offset,
                    'height': msg.roi.height,
                    'width': msg.roi.width,
                    'do_rectify': msg.roi.do_rectify
                }
            }

            # Save to JSON file
            json_path = os.path.join(self.output_dir, 'camera_info.json')
            with open(json_path, 'w') as f:
                json.dump(camera_info, f, indent=4)

            self.get_logger().info(f'Camera info saved to {json_path}')
            self.camera_info_received = True

    # Buffer callbacks to store the latest data
    def laserscan_buffer_callback(self, msg):
        """Buffer the latest laser scan message"""
        self.latest_laserscan = msg
        self.laserscan_frame_stamp = msg.header.stamp

    def pointcloud_buffer_callback(self, msg):
        """Buffer the latest pointcloud message"""
        self.latest_pointcloud = msg
        self.pointcloud_frame_stamp = msg.header.stamp

    def image_callback(self, msg):
        """Process image message (compressed or raw) and trigger other data processing"""
        # Always increment total frame count (total frames received)
        self.frame_count += 1

        if self.start_time is None:
            self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.video_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.get_logger().info(f'Recording started at {self.start_time}')

        if self.check_duration():
            if not self.is_shutdown:
                self.get_logger().info(f'Duration reached, stopping recording...')
                self.cleanup()
                self.is_shutdown = True
                # Schedule a final shutdown with a small delay to ensure cleanup completes
                threading.Timer(1.0, self.final_shutdown).start()
            return

        # Check if we should save this frame based on skip_frames setting
        if not self.should_save_frame():
            return  # Skip this frame

        # Increment saved frame count only when we actually save
        self.saved_frame_count += 1

        try:
            # Decode based on image type
            if self.image_type == 'compressed':
                # Decode the JPEG compressed data using PIL
                image_stream = io.BytesIO(msg.data)
                pil_image = PILImage.open(image_stream)
                numpy_image = np.array(pil_image)  # This will be RGB data from PIL
            elif self.image_type == 'raw':
                # Convert raw ROS Image message to numpy array using CvBridge
                # The encoding in the message tells us the format (e.g., 'bgr8', 'rgb8', 'mono8')
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                numpy_image = np.array(cv_image)
            else:
                self.get_logger().error(f'Unknown image type: {self.image_type}')
                return

        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {str(e)}")
            # Fallback method for compressed images
            if self.image_type == 'compressed':
                try:
                    numpy_image = imageio.imread(io.BytesIO(msg.data))
                    self.get_logger().info("Used imageio fallback for decoding")
                except Exception as e2:
                    self.get_logger().error(f"All image decoding methods failed: {e2}")
                    return
            else:
                return

        # Save frame using saved_frame_count for consistent numbering
        if self.record_frames:
            frame_path = os.path.join(self.frames_dir, f'frame_{self.saved_frame_count:06d}.jpg')
            # Save using PIL - no conversion needed as it's already in correct format for saving
            PILImage.fromarray(numpy_image).save(frame_path, "JPEG", quality=95)

        # For video writing without OpenCV:
        if self.record_video:
            # Store frame and its timestamp
            self.video_frames.append(numpy_image)
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.frame_timestamps.append(current_time - self.video_start_time)

        # Process synchronized laser scan data if available
        if self.record_laserscan and self.latest_laserscan is not None:
            self.process_laserscan(self.latest_laserscan, self.saved_frame_count)

        # Process synchronized pointcloud data if available
        if self.record_pointcloud and self.latest_pointcloud is not None:
            self.process_pointcloud(self.latest_pointcloud, self.saved_frame_count)

        # Log progress occasionally with frame skip info
        if self.saved_frame_count % 10 == 0:
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            if self.start_time is not None:
                elapsed = current_time - self.start_time
                skip_ratio = self.saved_frame_count / self.frame_count if self.frame_count > 0 else 0
                self.get_logger().info(
                    f'Saved frame {self.saved_frame_count} (received: {self.frame_count}, '
                    f'skip ratio: {skip_ratio:.2f}, elapsed: {elapsed:.1f}s)'
                )

    def process_laserscan(self, msg, frame_number):
        """Process laser scan message with frame synchronization"""
        try:
            # Extract laser scan data directly without conversion to cartesian
            ranges = np.array(msg.ranges)
            intensities = np.array(msg.intensities) if len(msg.intensities) > 0 else np.zeros_like(ranges)

            # Save raw data as numpy arrays
            data_path = os.path.join(self.laserscan_dir, f'scan_{frame_number:06d}.npz')
            np.savez(data_path,
                     ranges=ranges,
                     intensities=intensities,
                     angle_min=msg.angle_min,
                     angle_max=msg.angle_max,
                     angle_increment=msg.angle_increment)

            # Save additional metadata
            metadata = {
                'frame_id': msg.header.frame_id,
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'time_increment': msg.time_increment,
                'scan_time': msg.scan_time,
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'frame_number': frame_number  # Use the synchronized frame number
            }

            meta_path = os.path.join(self.laserscan_dir, f'scan_{frame_number:06d}_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            self.get_logger().error(f'Error processing laser scan for frame {frame_number}: {str(e)}')

    def process_pointcloud(self, msg, frame_number):
        """Process point cloud message with frame synchronization"""
        try:
            # Convert PointCloud2 to structured numpy array
            pc_structured = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

            if pc_structured.size == 0:  # If no valid points were found
                self.get_logger().warning(f'No valid point cloud points for frame {frame_number}')
                return

            # Create a regular array from the structured array by stacking the x, y, z columns
            if np.issubdtype(pc_structured.dtype, np.void):  # Structured array
                # Extract fields into separate arrays and stack them
                x = pc_structured['x']
                y = pc_structured['y']
                z = pc_structured['z']
                points = np.column_stack((x, y, z))
            else:  # Regular array (already in the right format)
                points = pc_structured

            # Save as numpy array
            npy_path = os.path.join(self.pointcloud_dir, f'cloud_{frame_number:06d}.npy')
            np.save(npy_path, points)

            # Also save as CSV for broader compatibility
            csv_path = os.path.join(self.pointcloud_dir, f'cloud_{frame_number:06d}.csv')
            np.savetxt(csv_path, points, delimiter=',', header='x,y,z', comments='')

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud for frame {frame_number}: {str(e)}')

    def cleanup(self):
        """Clean up resources and ensure proper shutdown"""
        if self.is_shutdown:
            return

        self.get_logger().info('Performing cleanup...')

        if hasattr(self, 'video_frames') and self.video_frames:
            video_path = os.path.join(self.output_dir, 'video.mp4')

            try:
                # Calculate actual FPS based on frame timestamps
                if hasattr(self, 'frame_timestamps') and len(self.frame_timestamps) > 1:
                    # Calculate actual FPS based on elapsed time and frame count
                    actual_duration = self.frame_timestamps[-1]
                    actual_fps = len(self.video_frames) / actual_duration
                    self.get_logger().info(
                        f'Actual recording duration: {actual_duration:.2f}s with {len(self.video_frames)} frames')
                    self.get_logger().info(f'Video FPS: {actual_fps:.2f}')

                    # Use the calculated FPS, with a reasonable default
                    fps = actual_fps if actual_fps > 0 else 30.0
                else:
                    fps = 30.0

                self.get_logger().info(f'Writing video with {len(self.video_frames)} frames at {fps} FPS...')

                # imageio expects RGB data, which is what we've stored
                imageio.mimsave(video_path, self.video_frames, fps=fps, quality=8)
                self.get_logger().info(f'Video saved to {video_path}')
            except Exception as e:
                self.get_logger().error(f"Failed to write video: {str(e)}")

            self.video_frames = []

        # Write summary metadata with actual duration and frame statistics
        actual_duration = 0
        if self.start_time is not None:
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            actual_duration = current_time - self.start_time

        summary = {
            'total_frames_received': self.frame_count,
            'total_frames_saved': self.saved_frame_count,
            'skip_frames_setting': self.skip_frames,
            'save_ratio': self.saved_frame_count / self.frame_count if self.frame_count > 0 else 0,
            'requested_duration': self.duration if self.duration is not None else 'unlimited',
            'actual_duration_seconds': actual_duration,
            'image_type': self.image_type,
            'image_topic': self.image_topic,
            'laserscan_topic': self.laserscan_topic,
            'pointcloud_topic': self.pointcloud_topic,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Add actual FPS calculation if we have frame timestamps
        if hasattr(self, 'frame_timestamps') and len(self.frame_timestamps) > 1:
            summary['video_fps'] = len(self.frame_timestamps) / self.frame_timestamps[-1]

        summary_path = os.path.join(self.output_dir, 'recording_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Close progress bar if it exists
        if self.progress_bar is not None:
            self.progress_bar.close()

        self.is_shutdown = True
        self.get_logger().info(
            f'Cleanup complete. Saved {self.saved_frame_count} frames out of {self.frame_count} received')

    def timer_callback(self):
        """Check if recording duration has elapsed and handle shutdown"""
        if self.check_duration() and not self.is_shutdown:
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            elapsed_time = current_time - self.start_time if self.start_time is not None else 0
            self.get_logger().info(f'Recording complete. Elapsed time: {elapsed_time:.2f}s')
            self.cleanup()

            # Schedule a final shutdown with a small delay to ensure cleanup completes
            threading.Timer(1.0, self.final_shutdown).start()

    def final_shutdown(self):
        """Perform final shutdown after cleanup"""
        self.get_logger().info('Shutting down node...')
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Record data from ROS2 topics')

    # Directory and timing parameters
    parser.add_argument('--output_dir', type=str, default='./recorded_data',
                        help='Output directory for recorded data')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Number of frames to skip between saves (0=save all, 1=save every other, 2=save every 3rd, etc.)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Recording duration in seconds (None = unlimited)')

    # Topic parameters
    parser.add_argument('--camera_info_topic', type=str, default='/drone1/camera/color/camera_info',
                        help='Topic for camera_info messages')
    parser.add_argument('--image_topic', type=str, default='/drone1/camera/color/compressed',
                        help='Topic for image messages')
    parser.add_argument('--image_type', type=str, default='compressed', choices=['compressed', 'raw'],
                        help='Type of image topic: "compressed" or "raw" (default: compressed)')
    parser.add_argument('--laserscan_topic', type=str, default='/drone1/lidar/laserscan',
                        help='Topic for laser scan messages')
    parser.add_argument('--pointcloud_topic', type=str, default='/drone1/lidar/pointcloud',
                        help='Topic for point cloud messages')

    # Recording options
    parser.add_argument('--no_video', action='store_false', dest='record_video',
                        help='Disable video recording')
    parser.add_argument('--no_frames', action='store_false', dest='record_frames',
                        help='Disable individual frame saving')
    parser.add_argument('--no_camera_info', action='store_false', dest='record_camera_info',
                        help='Disable camera info saving')
    parser.add_argument('--no_laserscan', action='store_false', dest='record_laserscan',
                        help='Disable laser scan saving')
    parser.add_argument('--no_pointcloud', action='store_false', dest='record_pointcloud',
                        help='Disable point cloud saving')
    parser.add_argument('--no_progress', action='store_false', dest='show_progress',
                        help='Disable progress bar')

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)

    # Initialize ROS2
    rclpy.init()

    # Create and run node
    recorder = DataRecorder(
        output_dir=output_dir,
        skip_frames=args.skip_frames,
        duration=args.duration,
        camera_info_topic=args.camera_info_topic,
        image_topic=args.image_topic,
        laserscan_topic=args.laserscan_topic,
        pointcloud_topic=args.pointcloud_topic,
        image_type=args.image_type,
        record_video=args.record_video,
        record_frames=args.record_frames,
        record_camera_info=args.record_camera_info,
        record_laserscan=args.record_laserscan,
        record_pointcloud=args.record_pointcloud,
        show_progress=args.show_progress
    )

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure cleanup is called
        if not recorder.is_shutdown:
            recorder.cleanup()
        recorder.destroy_node()


if __name__ == '__main__':
    main()