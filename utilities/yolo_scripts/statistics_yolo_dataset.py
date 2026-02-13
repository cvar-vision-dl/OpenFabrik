import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Import YAML if available
try:
    import yaml
except ImportError:
    yaml = None


def parse_yolo_segmentation_label(label_path, return_full_data=False):
    """Parse a YOLO segmentation label file and return class IDs."""
    instances = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:  # At least class_id and one coordinate pair
                    class_id = int(parts[0])
                    coordinates = [float(x) for x in parts[1:]]
                    num_points = len(coordinates) // 2

                    instance = {
                        'class_id': class_id,
                        'num_points': num_points
                    }

                    if return_full_data:
                        instance['coordinates'] = coordinates

                    instances.append(instance)
    except Exception as e:
        print(f"Warning: Could not parse {label_path}: {e}")
    return instances


def find_dataset_yaml(dataset_path):
    """Find and load class names from dataset.yaml or data.yaml in the dataset directory."""
    dataset_path = Path(dataset_path)

    # Common YAML file names for YOLO datasets
    yaml_names = ['dataset.yaml', 'data.yaml', 'config.yaml', 'dataset.yml', 'data.yml']

    for yaml_name in yaml_names:
        yaml_path = dataset_path / yaml_name
        if yaml_path.exists():
            if yaml is None:
                print(f"⚠️  Found '{yaml_name}' but PyYAML is not installed.")
                print("   Install it with: pip install pyyaml")
                return None

            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

                if data and 'names' in data:
                    names = data['names']
                    print(f"✅ Loaded class names from '{yaml_path.name}'")

                    # Handle different formats
                    if isinstance(names, dict):
                        # Convert to int keys if they're strings
                        return {int(k) if isinstance(k, (str, int)) else k: v for k, v in names.items()}
                    elif isinstance(names, list):
                        return {i: name for i, name in enumerate(names)}

            except Exception as e:
                print(f"⚠️  Error parsing '{yaml_path}': {e}")

    return None


def analyze_dataset(dataset_path, class_names=None, sample_images=20):
    """Analyze YOLO segmentation dataset structure and gather statistics."""
    dataset_path = Path(dataset_path)

    # Find available splits
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Dataset structure invalid. Expected 'images' and 'labels' folders in {dataset_path}")

    # Detect available splits
    splits = []
    for split_dir in images_dir.iterdir():
        if split_dir.is_dir():
            split_name = split_dir.name
            if (labels_dir / split_name).exists():
                splits.append(split_name)

    if not splits:
        print("No splits detected with matching image/label folders.")
        return None

    print(f"Found splits: {splits}")

    # Statistics storage
    stats = {
        'splits': splits,
        'class_counts': defaultdict(lambda: defaultdict(int)),  # split -> class_id -> count
        'image_counts': defaultdict(int),  # split -> count
        'total_instances': defaultdict(int),  # split -> count
        'images_per_class': defaultdict(lambda: defaultdict(list)),
        # split -> class_id -> list of (img_path, label_path)
        'polygon_points_stats': defaultdict(lambda: defaultdict(list)),  # split -> class_id -> list of point counts
        'empty_images': defaultdict(int),  # split -> count of images without labels
    }

    # Analyze each split
    for split in splits:
        split_labels_dir = labels_dir / split
        split_images_dir = images_dir / split

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(split_images_dir.glob(f'*{ext}'))
            image_files.extend(split_images_dir.glob(f'*{ext.upper()}'))

        stats['image_counts'][split] = len(image_files)

        # Process each image's label file
        for img_path in image_files:
            label_path = split_labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                stats['empty_images'][split] += 1
                continue

            instances = parse_yolo_segmentation_label(label_path)

            if not instances:
                stats['empty_images'][split] += 1
                continue

            stats['total_instances'][split] += len(instances)

            # Track unique classes in this image
            classes_in_image = set()

            for instance in instances:
                class_id = instance['class_id']
                num_points = instance['num_points']

                stats['class_counts'][split][class_id] += 1
                classes_in_image.add(class_id)
                stats['polygon_points_stats'][split][class_id].append(num_points)

            # Track which images contain each class
            for class_id in classes_in_image:
                # Store both image and label paths as absolute paths
                img_path_abs = str(img_path.absolute())
                label_path_abs = str(label_path.absolute())

                stats['images_per_class'][split][class_id].append({
                    'image': img_path_abs,
                    'label': label_path_abs
                })

    # Sample random images per class
    import random
    stats['sample_images'] = {}

    for split in stats['images_per_class']:
        stats['sample_images'][split] = {}
        for class_id in stats['images_per_class'][split]:
            images = stats['images_per_class'][split][class_id]
            # Sample random images (or all if fewer than requested)
            num_samples = min(sample_images, len(images))
            sampled = random.sample(images, num_samples)

            # Ensure all paths are strings
            for item in sampled:
                if not isinstance(item['image'], str):
                    item['image'] = str(item['image'])
                if not isinstance(item['label'], str):
                    item['label'] = str(item['label'])

            stats['sample_images'][split][class_id] = sampled

    # Convert image lists to counts for the main statistics
    stats['images_per_class_count'] = {}
    for split in stats['images_per_class']:
        stats['images_per_class_count'][split] = {}
        for class_id in stats['images_per_class'][split]:
            stats['images_per_class_count'][split][class_id] = len(stats['images_per_class'][split][class_id])

    # Calculate polygon point statistics (average per class)
    stats['avg_polygon_points'] = {}
    for split in stats['polygon_points_stats']:
        stats['avg_polygon_points'][split] = {}
        for class_id in stats['polygon_points_stats'][split]:
            points = stats['polygon_points_stats'][split][class_id]
            if points:
                stats['avg_polygon_points'][split][class_id] = sum(points) / len(points)

    # Prepare data for HTML
    stats['class_names'] = class_names
    stats['dataset_path'] = str(dataset_path)
    stats['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return stats


def generate_html_report(stats, output_path='dataset_analysis.html'):
    """Generate an interactive HTML report with embedded Chart.js."""

    # Convert absolute paths to relative paths from the HTML file location
    output_path_abs = Path(output_path).absolute()
    output_dir = output_path_abs.parent

    print(f"\n📁 HTML will be saved to: {output_path_abs}")
    print(f"📁 HTML directory: {output_dir}")

    # Convert image and label paths to relative
    paths_converted = 0
    paths_failed = 0

    for split in stats['sample_images']:
        for class_id in stats['sample_images'][split]:
            for img_data in stats['sample_images'][split][class_id]:
                img_path_str = img_data['image']
                label_path_str = img_data['label']

                try:
                    img_path = Path(img_path_str).resolve()
                    label_path = Path(label_path_str).resolve()

                    # Calculate relative path from HTML location to image
                    img_relative = os.path.relpath(str(img_path), str(output_dir))
                    label_relative = os.path.relpath(str(label_path), str(output_dir))

                    # Ensure forward slashes for URLs and no leading slash
                    img_relative = img_relative.replace('\\', '/')
                    label_relative = label_relative.replace('\\', '/')

                    # Remove any leading slash that might cause file:// issues
                    if img_relative.startswith('/'):
                        img_relative = img_relative.lstrip('/')
                    if label_relative.startswith('/'):
                        label_relative = label_relative.lstrip('/')

                    img_data['image'] = img_relative
                    img_data['label'] = label_relative
                    paths_converted += 1

                    if paths_converted == 1:  # Debug: print first conversion
                        print(f"\n🔍 Sample path conversion:")
                        print(f"   From: {img_path}")
                        print(f"   To:   {img_relative}")

                except (ValueError, TypeError) as e:
                    print(f"⚠️  Warning: Could not create relative path for {img_path_str}: {e}")
                    paths_failed += 1
                    # Keep original path

    print(f"\n✅ Converted {paths_converted} image paths to relative paths")
    if paths_failed > 0:
        print(f"⚠️  Failed to convert {paths_failed} paths")

    # Debug: Check a few sample paths after conversion
    print(f"\n🔍 Verifying sample paths in data structure:")
    for split in list(stats['sample_images'].keys())[:1]:  # Check first split
        for class_id in list(stats['sample_images'][split].keys())[:1]:  # Check first class
            samples = stats['sample_images'][split][class_id][:3]  # Check first 3 images
            for i, img_data in enumerate(samples):
                print(f"   Sample {i + 1}: {img_data['image']}")

    # Get all unique class IDs across all splits
    all_class_ids = set()
    for split in stats['class_counts']:
        all_class_ids.update(stats['class_counts'][split].keys())
    all_class_ids = sorted(all_class_ids)

    # Prepare class names
    if stats.get('class_names'):
        class_labels = [stats['class_names'].get(cid, f'Class {cid}') for cid in all_class_ids]
    else:
        class_labels = [f'Class {cid}' for cid in all_class_ids]

    # Prepare data for charts
    splits = stats['splits']

    # Data for instances per class per split
    datasets_per_split = []
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']

    for i, split in enumerate(splits):
        data = [stats['class_counts'][split].get(cid, 0) for cid in all_class_ids]
        datasets_per_split.append({
            'label': split,
            'data': data,
            'backgroundColor': colors[i % len(colors)],
        })

    # Overall statistics
    total_images = sum(stats['image_counts'].values())
    total_instances = sum(stats['total_instances'].values())
    total_empty = sum(stats['empty_images'].values())

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Segmentation Dataset Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .info-box {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 30px;
            border-radius: 5px;
        }}
        .info-box p {{
            margin: 5px 0;
            color: #555;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .chart-container {{
            position: relative;
            margin-bottom: 50px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .chart-title {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            color: #555;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .section {{
            margin-top: 50px;
        }}

        /* Image Gallery Styles */
        .class-gallery {{
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            background: white;
        }}
        .gallery-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}
        .gallery-header:hover {{
            opacity: 0.9;
        }}
        .gallery-header h3 {{
            margin: 0;
            font-size: 1.2em;
        }}
        .gallery-toggle {{
            font-size: 1.5em;
            transition: transform 0.3s;
        }}
        .gallery-toggle.open {{
            transform: rotate(180deg);
        }}
        .gallery-content {{
            display: none;
            padding: 20px;
        }}
        .gallery-content.open {{
            display: block;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .image-container {{
            position: relative;
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .image-container:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }}
        .image-container img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .image-container canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}

        /* Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            overflow: auto;
        }}
        .modal.open {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .modal-content {{
            position: relative;
            max-width: 90%;
            max-height: 90%;
            margin: auto;
        }}
        .modal-content img {{
            max-width: 100%;
            max-height: 90vh;
            display: block;
        }}
        .modal-content canvas {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
        }}
        .modal-close:hover {{
            color: #ccc;
        }}
        .split-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .split-tab {{
            padding: 10px 20px;
            background: #f0f0f0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }}
        .split-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .split-tab:hover {{
            background: #e0e0e0;
        }}
        .split-tab.active:hover {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .split-content {{
            display: none;
        }}
        .split-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 YOLO Segmentation Dataset Analysis</h1>
        <div class="subtitle">Generated on {stats['analysis_date']}</div>

        <div class="info-box">
            <p><strong>Dataset Path:</strong> {stats['dataset_path']}</p>
            <p><strong>Splits Found:</strong> {', '.join(stats['splits'])}</p>
            <p><strong>Total Classes:</strong> {len(all_class_ids)}</p>
            <p style="color: #e67e22; margin-top: 10px;"><strong>⚠️ Note:</strong> Serve this HTML file via HTTP server from the same directory for images to load correctly.</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{total_images}</div>
                <div class="label">Total Images</div>
            </div>
            <div class="stat-card">
                <div class="number">{total_instances}</div>
                <div class="label">Total Instances</div>
            </div>
            <div class="stat-card">
                <div class="number">{len(all_class_ids)}</div>
                <div class="label">Unique Classes</div>
            </div>
            <div class="stat-card">
                <div class="number">{total_empty}</div>
                <div class="label">Empty Images</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Instances per Class (by Split)</div>
            <canvas id="classDistributionChart"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">Total Instances per Split</div>
            <canvas id="splitDistributionChart"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">Images per Split</div>
            <canvas id="imageDistributionChart"></canvas>
        </div>

        <div class="section">
            <h2 style="margin-bottom: 20px; color: #333;">📋 Detailed Class Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class ID</th>
                        <th>Class Name</th>
                        {' '.join([f'<th>{split} Instances</th>' for split in splits])}
                        <th>Total Instances</th>
                    </tr>
                </thead>
                <tbody id="classTable">
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 style="margin-bottom: 20px; color: #333;">🖼️ Sample Images per Class</h2>
            <p style="color: #666; margin-bottom: 20px;">Click on images to view full size with segmentation overlay</p>

            <div class="split-tabs" id="splitTabs">
                {' '.join([f'<button class="split-tab" data-split="{split}">{split}</button>' for split in splits])}
            </div>

            <div id="imageGalleryContainer">
            </div>
        </div>

        <!-- Modal for full-size image -->
        <div id="imageModal" class="modal">
            <span class="modal-close">&times;</span>
            <div class="modal-content" id="modalContent">
            </div>
        </div>
    </div>

    <script>
        // Wait for DOM to be fully loaded
        document.addEventListener('DOMContentLoaded', function() {{
            initializeAnalysis();
        }});

        function initializeAnalysis() {{
            const classLabels = {json.dumps(class_labels)};
            const classIds = {json.dumps(all_class_ids)};
            const splits = {json.dumps(splits)};
            const classCounts = {json.dumps(dict(stats['class_counts']))};
            const imageCounts = {json.dumps(dict(stats['image_counts']))};
            const totalInstances = {json.dumps(dict(stats['total_instances']))};
            const sampleImages = {json.dumps(stats['sample_images'])};

            // Debug: Check what paths we received
            console.log('=== Dataset Analysis Debug Info ===');
            console.log('Current page URL:', window.location.href);
            console.log('Number of splits:', splits.length);

            // Show sample image paths from first split and class
            if (splits.length > 0) {{
                const firstSplit = splits[0];
                console.log('First split:', firstSplit);
                const classes = Object.keys(sampleImages[firstSplit] || {{}});
                console.log('Classes in first split:', classes);

                if (classes.length > 0) {{
                    const firstClass = classes[0];
                    const images = sampleImages[firstSplit][firstClass];
                    console.log(`Images in first class (${{firstClass}}):`, images.length);

                    // Show first 3 image paths
                    for (let i = 0; i < Math.min(3, images.length); i++) {{
                        console.log(`  Image ${{i+1}}: ${{images[i].image}}`);
                        console.log(`  Label ${{i+1}}: ${{images[i].label}}`);
                    }}
                }}
            }}
            console.log('===================================');

            // Make sure all required elements exist
            const requiredElements = ['imageGalleryContainer', 'splitTabs', 'imageModal', 'modalContent'];
            const missingElements = [];

            requiredElements.forEach(id => {{
                if (!document.getElementById(id)) {{
                    missingElements.push(id);
                }}
            }});

            if (missingElements.length > 0) {{
                console.error('ERROR: Missing required elements:', missingElements);
                return;
            }}


        // Parse label file content
        function parseLabelFile(content) {{
            const lines = content.trim().split('\\n');
            const instances = [];
            for (const line of lines) {{
                if (!line.trim()) continue;
                const parts = line.trim().split(/\\s+/);
                if (parts.length >= 3) {{
                    const classId = parseInt(parts[0]);
                    const coordinates = parts.slice(1).map(x => parseFloat(x));
                    instances.push({{ classId, coordinates }});
                }}
            }}
            return instances;
        }}

        // Draw segmentation mask on canvas
        function drawSegmentation(canvas, img, labelPath, targetClassId) {{
            const ctx = canvas.getContext('2d');
            canvas.width = img.naturalWidth || img.width;
            canvas.height = img.naturalHeight || img.height;

            // Fetch and parse label file
            fetch(labelPath)
                .then(response => {{
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    return response.text();
                }})
                .then(content => {{
                    const instances = parseLabelFile(content);

                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    instances.forEach(instance => {{
                        if (instance.classId === targetClassId) {{
                            const coords = instance.coordinates;
                            if (coords.length < 4) return;

                            // Draw filled polygon
                            ctx.fillStyle = 'rgba(102, 126, 234, 0.4)';
                            ctx.strokeStyle = 'rgba(102, 126, 234, 0.9)';
                            ctx.lineWidth = 2;

                            ctx.beginPath();
                            for (let i = 0; i < coords.length; i += 2) {{
                                const x = coords[i] * canvas.width;
                                const y = coords[i + 1] * canvas.height;
                                if (i === 0) {{
                                    ctx.moveTo(x, y);
                                }} else {{
                                    ctx.lineTo(x, y);
                                }}
                            }}
                            ctx.closePath();
                            ctx.fill();
                            ctx.stroke();
                        }}
                    }});
                }})
                .catch(err => {{
                    console.error('Error loading label file:', err);
                    drawSegmentationFallback(canvas, labelPath, targetClassId);
                }});
        }}

        // Fallback method that doesn't rely on fetch
        function drawSegmentationFallback(canvas, labelPath, targetClassId) {{
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Overlay Unavailable', canvas.width / 2, canvas.height / 2);
            ctx.font = '10px Arial';
            ctx.fillText('(Check HTTP server location)', canvas.width / 2, canvas.height / 2 + 20);
        }}        
        // Create image gallery for each class and split
        function createImageGalleries() {{
            const container = document.getElementById('imageGalleryContainer');

            splits.forEach(split => {{
                const splitDiv = document.createElement('div');
                splitDiv.className = 'split-content';
                splitDiv.id = `split-${{split}}`;

                classIds.forEach((classId, index) => {{
                    const images = sampleImages[split]?.[classId] || [];
                    if (images.length === 0) return;

                    const galleryDiv = document.createElement('div');
                    galleryDiv.className = 'class-gallery';

                    const header = document.createElement('div');
                    header.className = 'gallery-header';
                    header.innerHTML = `
                        <h3>${{classLabels[index]}} (Class ${{classId}}) - ${{images.length}} images</h3>
                        <span class="gallery-toggle">▼</span>
                    `;

                    const content = document.createElement('div');
                    content.className = 'gallery-content';

                    const grid = document.createElement('div');
                    grid.className = 'image-grid';

                    images.forEach(imgData => {{
                        const imgContainer = document.createElement('div');
                        imgContainer.className = 'image-container';

                        const img = document.createElement('img');
                        img.src = imgData.image;
                        img.alt = `Class ${{classId}} sample`;
                        img.onerror = function() {{
                            // Fallback for file protocol issues
                            this.style.display = 'none';
                            imgContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;background:#f0f0f0;color:#666;">Image not accessible</div>';
                        }};

                        const canvas = document.createElement('canvas');

                        img.onload = function() {{
                            drawSegmentation(canvas, img, imgData.label, classId);
                        }};

                        imgContainer.appendChild(img);
                        imgContainer.appendChild(canvas);

                        imgContainer.onclick = function() {{
                            openModal(imgData.image, imgData.label, classId);
                        }};

                        grid.appendChild(imgContainer);
                    }});

                    content.appendChild(grid);
                    galleryDiv.appendChild(header);
                    galleryDiv.appendChild(content);
                    splitDiv.appendChild(galleryDiv);

                    // Toggle functionality
                    header.onclick = function() {{
                        content.classList.toggle('open');
                        header.querySelector('.gallery-toggle').classList.toggle('open');
                    }};
                }});

                container.appendChild(splitDiv);
            }});
        }}

        // Modal functionality
        function openModal(imagePath, labelPath, classId) {{
            const modal = document.getElementById('imageModal');
            const modalContent = document.getElementById('modalContent');

            modalContent.innerHTML = '';

            const img = document.createElement('img');
            img.src = imagePath;

            const canvas = document.createElement('canvas');

            img.onload = function() {{
                // Wait for the image to be displayed and get its actual rendered size
                setTimeout(() => {{
                    // Set canvas to match the DISPLAYED size of the image
                    canvas.width = img.clientWidth;
                    canvas.height = img.clientHeight;
                    canvas.style.width = img.clientWidth + 'px';
                    canvas.style.height = img.clientHeight + 'px';

                    // Now draw segmentation at the correct scale
                    const ctx = canvas.getContext('2d');

                    fetch(labelPath)
                        .then(response => {{
                            if (!response.ok) {{
                                throw new Error(`HTTP error! status: ${{response.status}}`);
                            }}
                            return response.text();
                        }})
                        .then(content => {{
                            const instances = parseLabelFile(content);

                            ctx.clearRect(0, 0, canvas.width, canvas.height);

                            instances.forEach(instance => {{
                                if (instance.classId === classId) {{
                                    const coords = instance.coordinates;
                                    if (coords.length < 4) return;

                                    // Draw filled polygon
                                    ctx.fillStyle = 'rgba(102, 126, 234, 0.4)';
                                    ctx.strokeStyle = 'rgba(102, 126, 234, 0.9)';
                                    ctx.lineWidth = 2;

                                    ctx.beginPath();
                                    for (let i = 0; i < coords.length; i += 2) {{
                                        const x = coords[i] * canvas.width;
                                        const y = coords[i + 1] * canvas.height;
                                        if (i === 0) {{
                                            ctx.moveTo(x, y);
                                        }} else {{
                                            ctx.lineTo(x, y);
                                        }}
                                    }}
                                    ctx.closePath();
                                    ctx.fill();
                                    ctx.stroke();
                                }}
                            }});
                        }})
                        .catch(err => {{
                            console.error('Error loading label file in modal:', err);
                        }});
                }}, 50); // Small delay to ensure image is rendered
            }};

            modalContent.appendChild(img);
            modalContent.appendChild(canvas);
            modal.classList.add('open');
        }}

        function closeModal() {{
            document.getElementById('imageModal').classList.remove('open');
        }}

        // Close modal when clicking outside or on X
        document.getElementById('imageModal').onclick = function(event) {{
            if (event.target === this) {{
                closeModal();
            }}
        }};

        const closeBtn = document.querySelector('.modal-close');
        if (closeBtn) {{
            closeBtn.onclick = closeModal;
        }}

        // Chart 1: Instances per class
        const ctx1 = document.getElementById('classDistributionChart').getContext('2d');
        new Chart(ctx1, {{
            type: 'bar',
            data: {{
                labels: classLabels,
                datasets: {json.dumps(datasets_per_split)}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Number of Instances'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Class'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }}
            }}
        }});

        // Chart 2: Total instances per split
        const ctx2 = document.getElementById('splitDistributionChart').getContext('2d');
        new Chart(ctx2, {{
            type: 'doughnut',
            data: {{
                labels: splits,
                datasets: [{{
                    data: splits.map(s => totalInstances[s]),
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});

        // Chart 3: Images per split
        const ctx3 = document.getElementById('imageDistributionChart').getContext('2d');
        new Chart(ctx3, {{
            type: 'bar',
            data: {{
                labels: splits,
                datasets: [{{
                    label: 'Number of Images',
                    data: splits.map(s => imageCounts[s]),
                    backgroundColor: '#667eea'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Number of Images'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});

        // Populate detailed table
        const tableBody = document.getElementById('classTable');
        classIds.forEach((classId, index) => {{
            const row = document.createElement('tr');
            let totalCount = 0;

            let rowHTML = `<td>${{classId}}</td><td>${{classLabels[index]}}</td>`;

            splits.forEach(split => {{
                const count = classCounts[split][classId] || 0;
                totalCount += count;
                rowHTML += `<td>${{count}}</td>`;
            }});

            rowHTML += `<td><strong>${{totalCount}}</strong></td>`;
            row.innerHTML = rowHTML;
            tableBody.appendChild(row);
        }});

        // Create image galleries FIRST
        createImageGalleries();

        // THEN set up split tabs functionality
        document.querySelectorAll('.split-tab').forEach(tab => {{
            tab.onclick = function() {{
                const split = this.getAttribute('data-split');
                console.log('Switching to split:', split);

                // Remove active class from all tabs and contents
                document.querySelectorAll('.split-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.split-content').forEach(c => c.classList.remove('active'));

                // Add active class to clicked tab and corresponding content
                this.classList.add('active');
                const splitContent = document.getElementById(`split-${{split}}`);
                if (splitContent) {{
                    splitContent.classList.add('active');
                    console.log('Activated split content for:', split);
                }} else {{
                    console.error('Split content not found for:', split);
                    console.log('Available split elements:', Array.from(document.querySelectorAll('.split-content')).map(el => el.id));
                }}
            }};
        }});

        // Activate first split by default
        if (splits.length > 0) {{
            document.querySelector('.split-tab').classList.add('active');
            document.getElementById(`split-${{splits[0]}}`).classList.add('active');
        }}

        console.log('Analysis initialization complete');
    }}
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    output_path_abs = Path(output_path).absolute()
    output_dir = output_path_abs.parent
    dataset_path = Path(stats['dataset_path']).absolute()

    print(f"\n✅ HTML report generated: {output_path}")
    print(f"\n" + "=" * 70)
    print("📌 IMPORTANT: How to view the report with images")
    print("=" * 70)
    print(f"\n1. The HTML file is saved at:")
    print(f"   {output_path_abs}")
    print(f"\n2. Your dataset is at:")
    print(f"   {dataset_path}")

    # Determine best directory to run server from
    try:
        # Try to find common parent directory
        common_parent = output_dir
        rel_check = os.path.relpath(dataset_path, output_dir)
        if rel_check.startswith('..'):
            # Dataset is not under output_dir, need to go up
            # Find a common ancestor
            all_parents_output = [output_dir] + list(output_dir.parents)
            all_parents_dataset = [dataset_path] + list(dataset_path.parents)

            for parent in all_parents_output:
                if parent in all_parents_dataset:
                    common_parent = parent
                    break

        print(f"\n3. Run the HTTP server from this directory:")
        print(f"   cd {common_parent}")
        print(f"   python -m http.server 8000")

        # Calculate relative path from common parent to HTML file
        html_rel = os.path.relpath(output_path_abs, common_parent)
        print(f"\n4. Open this URL in your browser:")
        print(f"   http://localhost:8000/{html_rel.replace(os.sep, '/')}")

    except Exception as e:
        print(f"\n3. Run the HTTP server from a directory that contains both:")
        print(f"   - The HTML file")
        print(f"   - The dataset directory")
        print(f"\n   Then run: python -m http.server 8000")

    print("\n" + "=" * 70)
    print("\n💡 TIP: For easier access, consider saving the HTML inside your")
    print(f"   dataset directory with: -o {dataset_path / 'analysis.html'}")
    print("=" * 70)


def load_class_names(class_file):
    """Load class names from a file. Supports .txt, .names, or .json formats."""
    class_file = Path(class_file)

    if not class_file.exists():
        print(f"Warning: Class file '{class_file}' not found. Using auto-generated names.")
        return None

    try:
        if class_file.suffix == '.json':
            with open(class_file, 'r') as f:
                data = json.load(f)
                # Support both list and dict formats
                if isinstance(data, list):
                    return {i: name for i, name in enumerate(data)}
                elif isinstance(data, dict):
                    return {int(k): v for k, v in data.items()}
        else:  # .txt or .names
            with open(class_file, 'r') as f:
                names = [line.strip() for line in f if line.strip()]
                return {i: name for i, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not parse class file '{class_file}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Analyze YOLO segmentation dataset and generate interactive HTML report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dataset
  %(prog)s /path/to/dataset -o my_report.html
  %(prog)s /path/to/dataset -c classes.txt
  %(prog)s /path/to/dataset -o report.html -c classes.txt -s 30
  %(prog)s /path/to/dataset --sample-images 50

Notes:
  - If -c/--classes is not provided, the script will automatically search for 
    dataset.yaml or data.yaml in the dataset directory and extract class names.
  - If PyYAML is not installed, you can install it with: pip install pyyaml
        """
    )

    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to the YOLO dataset directory (containing images/ and labels/ folders)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='dataset_analysis.html',
        help='Output HTML file path (default: dataset_analysis.html). Tip: Save inside your dataset directory for easier access.'
    )

    parser.add_argument(
        '-c', '--classes',
        type=str,
        default=None,
        help='Path to class names file (.txt, .names, or .json). If not provided, will search for dataset.yaml/data.yaml automatically.'
    )

    parser.add_argument(
        '-s', '--sample-images',
        type=int,
        default=20,
        help='Number of random sample images to display per class (default: 20)'
    )

    args = parser.parse_args()

    # Load class names
    class_names = None
    if args.classes:
        # User provided a class file explicitly
        class_names = load_class_names(args.classes)
    else:
        # Try to find dataset.yaml automatically
        print("No class file specified. Searching for dataset.yaml/data.yaml...")
        class_names = find_dataset_yaml(args.dataset_path)
        if class_names is None:
            print("⚠️  No dataset.yaml found. Using auto-generated class names (Class 0, Class 1, ...).")

    print("\n🔍 Analyzing YOLO segmentation dataset...")
    print(f"Dataset path: {args.dataset_path}")
    if class_names:
        print(f"Loaded {len(class_names)} class names")
    print(f"Sample images per class: {args.sample_images}")
    print(f"Output: {args.output}\n")

    try:
        stats = analyze_dataset(args.dataset_path, class_names, args.sample_images)

        if stats:
            # Print summary
            print("\n" + "=" * 50)
            print("SUMMARY")
            print("=" * 50)
            print(f"Splits: {', '.join(stats['splits'])}")
            print(f"Total images: {sum(stats['image_counts'].values())}")
            print(f"Total instances: {sum(stats['total_instances'].values())}")
            print(f"Empty images: {sum(stats['empty_images'].values())}")
            print("=" * 50 + "\n")

            # Generate HTML report
            generate_html_report(stats, args.output)
            print(f"\n🎉 Analysis complete! Open '{args.output}' in your browser to view the interactive report.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()