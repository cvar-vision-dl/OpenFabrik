#!/usr/bin/env python3
"""
Enhanced Model Performance and Speed Benchmark Analyzer
Loads all models in a folder, computes metrics and inference speed,
generates performance reports with interactive visualizations.
"""

import argparse
import os
import sys
import json
import time
import random
import re
import gc
import colorsys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from PIL import Image
    import torch
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, Input, Output
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install ultralytics matplotlib seaborn pandas pillow torch plotly dash")
    sys.exit(1)


class EnhancedModelVisualizer:
    """Enhanced visualizer with support for 100+ unique model combinations"""

    def __init__(self):
        # Generate 50 distinct colors using HSV color space
        self.colors = self._generate_distinct_colors(50)

        # Extended marker set with 30+ unique shapes
        self.markers = [
            'o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x', '8',
            'P', 'X', '1', '2', '3', '4', 'd', '|', '_', '.', ',', '0', '5', '6', '7', '9',
            # Custom shapes for even more variety
            '$\\bigodot$', '$\\bigoplus$', '$\\bigotimes$', '$\\bigcirc$', '$\\blacksquare$',
            '$\\blacktriangle$', '$\\blacktriangledown$', '$\\diamond$', '$\\star$',
            '$\\heartsuit$', '$\\spadesuit$', '$\\clubsuit$', '$\\diamondsuit$'
        ]

        # For plotly (different marker symbols)
        self.plotly_markers = [
            'circle', 'square', 'triangle-up', 'triangle-down', 'triangle-left',
            'triangle-right', 'diamond', 'cross', 'x', 'star', 'pentagon',
            'hexagon', 'octagon', 'star-triangle-up', 'star-triangle-down',
            'star-square', 'star-diamond', 'diamond-tall', 'diamond-wide',
            'hourglass', 'bowtie', 'circle-cross', 'circle-x', 'square-cross',
            'square-x', 'diamond-cross', 'diamond-x', 'cross-thin', 'x-thin',
            'asterisk', 'hash', 'y-up', 'y-down', 'y-left', 'y-right',
            'line-ew', 'line-ns', 'line-ne', 'line-nw'
        ]

    def _generate_distinct_colors(self, n):
        """Generate n visually distinct colors using HSV color space"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.15  # Vary brightness slightly
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})')
        return colors

    def create_interactive_plots(self, df, output_dir, task='pose'):
        """Create interactive plotly visualizations"""

        metrics_to_plot = {
            'map50_95': 'mAP@0.5:0.95',
            'map50': 'mAP@0.5',
            'precision': 'Precision',
            'recall': 'Recall'
        }

        if task == 'pose':
            metrics_to_plot = {k: v + ' (Keypoints)' for k, v in metrics_to_plot.items()}
        else:
            metrics_to_plot = {k: v + ' (Boxes)' for k, v in metrics_to_plot.items()}

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(metrics_to_plot.values()),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Add traces for each model
        for idx, (_, row) in enumerate(df.iterrows()):
            color = self.colors[idx % len(self.colors)]
            marker = self.plotly_markers[idx % len(self.plotly_markers)]

            # Hover text with all model info
            hover_text = (
                f"Model: {row['model_name']}<br>"
                f"Resolution: {row['resolution']}px<br>"
                f"Format: {row['file_extension']}<br>"
                f"Size: {row['file_size_mb']:.1f} MB<br>"
                f"FPS: {row['fps']:.1f}<br>"
                f"Inference: {row['inference_time_ms']:.1f} ms"
            )

            # Add to each subplot
            for i, metric in enumerate(metrics_to_plot.keys()):
                row_idx = i // 2 + 1
                col_idx = i % 2 + 1

                fig.add_trace(
                    go.Scatter(
                        x=[row['fps']],
                        y=[row[metric]],
                        mode='markers',
                        marker=dict(
                            symbol=marker,
                            size=12,
                            color=color,
                            line=dict(width=2, color='black')
                        ),
                        name=row['model_name'],
                        hovertemplate=hover_text + f"<br>{metrics_to_plot[metric]}: {row[metric]:.4f}<extra></extra>",
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row_idx, col=col_idx
                )

        # Update layout
        fig.update_layout(
            title=f'Interactive Model Performance Analysis - {task.title()} Task',
            height=800,
            width=1400,
            hovermode='closest'
        )

        # Update axes labels
        for i in range(2):
            for j in range(2):
                fig.update_xaxes(title_text="Frames Per Second (FPS)", row=i + 1, col=j + 1)

        # Save interactive plot
        fig.write_html(f"{output_dir}/interactive_performance_analysis.html")
        print(f"Interactive plot saved: {output_dir}/interactive_performance_analysis.html")

        return fig

    def create_main_performance_plot(self, df, output_dir):
        """Create main performance plot with hover functionality"""

        fig = go.Figure()

        for idx, (_, row) in enumerate(df.iterrows()):
            color = self.colors[idx % len(self.colors)]
            marker = self.plotly_markers[idx % len(self.plotly_markers)]

            hover_text = (
                f"<b>{row['model_name']}</b><br>"
                f"Resolution: {row['resolution']}px<br>"
                f"Format: {row['file_extension']}<br>"
                f"Size: {row['file_size_mb']:.1f} MB<br>"
                f"FPS: {row['fps']:.1f}<br>"
                f"Inference: {row['inference_time_ms']:.1f} ms<br>"
                f"mAP@0.5:0.95: {row['map50_95']:.4f}"
            )

            fig.add_trace(
                go.Scatter(
                    x=[row['fps']],
                    y=[row['map50_95']],
                    mode='markers',
                    marker=dict(
                        symbol=marker,
                        size=12,  # Fixed size for better visibility
                        color=color,
                        line=dict(width=2, color='black'),
                        opacity=0.8
                    ),
                    name=row['model_name'],
                    hovertemplate=hover_text + "<extra></extra>"
                )
            )

        fig.update_layout(
            title='Model Performance: Speed vs Accuracy<br><sub>Hover for details • Each point represents one model</sub>',
            xaxis_title='Frames Per Second (FPS)',
            yaxis_title='mAP@0.5:0.95',
            width=1200,
            height=700,
            hovermode='closest',
            showlegend=False  # Too many models for useful legend
        )

        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['fps'], df['map50_95'], 1)
            x_trend = np.linspace(df['fps'].min(), df['fps'].max(), 100)
            y_trend = np.polyval(z, x_trend)

            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    line=dict(dash='dash', color='rgba(0,0,0,0.5)', width=2),
                    name='Trend Line',
                    hoverinfo='skip'
                )
            )

        fig.write_html(f"{output_dir}/main_performance_plot.html")
        print(f"Main performance plot saved: {output_dir}/main_performance_plot.html")

        return fig

    def create_enhanced_static_plots(self, df, output_dir, task='pose'):
        """Create enhanced static plots with maximum color/shape variety"""

        # Create comprehensive color-shape combinations (50 colors × 30+ markers = 1500+ combinations)
        color_shape_combos = []
        for i in range(len(df)):
            color_idx = i % len(self.colors)
            marker_idx = i % len(self.markers)
            color_shape_combos.append((self.colors[color_idx], self.markers[marker_idx]))

        metrics_to_plot = {
            'map50_95': f'mAP@0.5:0.95 ({task.title()})',
            'map50': f'mAP@0.5 ({task.title()})',
            'precision': f'Precision ({task.title()})',
            'recall': f'Recall ({task.title()})'
        }

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        for idx, (metric, title) in enumerate(metrics_to_plot.items()):
            ax = axes[idx]

            # Plot each model with unique color-shape combination
            for i, (_, row) in enumerate(df.iterrows()):
                color = self.colors[i % len(self.colors)]
                marker = self.markers[i % len(self.markers)]

                # Convert RGB string to matplotlib color
                if color.startswith('rgb'):
                    # Extract RGB values and normalize
                    rgb_vals = color.replace('rgb(', '').replace(')', '').split(',')
                    color = [int(val.strip()) / 255.0 for val in rgb_vals]

                ax.scatter(row['fps'], row[metric],
                           c=[color], marker=marker, s=120, alpha=0.8,
                           edgecolors='black', linewidth=1)

            ax.set_xlabel('Frames Per Second (FPS)', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} vs FPS', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(df) > 1:
                z = np.polyfit(df['fps'], df[metric], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['fps'].min(), df['fps'].max(), 100)
                ax.plot(x_trend, p(x_trend), "k--", alpha=0.6, linewidth=2)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/enhanced_static_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create a searchable reference chart
        self._create_model_reference_chart(df, output_dir)

        print(f"Enhanced static plots saved to {output_dir}")

    def _create_model_reference_chart(self, df, output_dir):
        """Create a searchable reference chart for model identification"""

        # Calculate grid dimensions
        n_models = len(df)
        cols = min(10, max(8, int(np.sqrt(n_models))))
        rows = (n_models + cols - 1) // cols

        fig, ax = plt.subplots(figsize=(max(16, cols * 2), max(12, rows * 1.5)))

        for i, (_, row) in enumerate(df.iterrows()):
            col = i % cols
            row_idx = i // cols

            x = col
            y = rows - row_idx - 1  # Flip y to start from top

            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]

            # Convert RGB string to matplotlib color
            if color.startswith('rgb'):
                rgb_vals = color.replace('rgb(', '').replace(')', '').split(',')
                color = [int(val.strip()) / 255.0 for val in rgb_vals]

            ax.scatter(x, y, c=[color], marker=marker, s=400,
                       alpha=0.9, edgecolors='black', linewidth=2)

            # Add model info text
            ax.text(x, y - 0.15, row['model_name'], ha='center', va='top',
                    fontsize=8, weight='bold', wrap=True)
            ax.text(x, y - 0.25, f"{row['resolution']}px | {row['file_extension']}",
                    ha='center', va='top', fontsize=7, style='italic')
            ax.text(x, y - 0.35, f"mAP: {row['map50_95']:.3f} | {row['fps']:.1f} FPS",
                    ha='center', va='top', fontsize=6, alpha=0.8)

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_title('Model Reference Guide\n(Use this to identify models in other plots)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_reference_guide.png', dpi=300, bbox_inches='tight')
        plt.close()


class OfflineModelAnalyzer:
    """Offline interactive analyzer using Dash"""

    def __init__(self, benchmark_json_path):
        self.data_path = benchmark_json_path
        self.df = self.load_data()
        self.colors = self._generate_distinct_colors(len(self.df))

        # Plotly marker symbols
        self.markers = [
            'circle', 'square', 'triangle-up', 'triangle-down', 'triangle-left',
            'triangle-right', 'diamond', 'cross', 'x', 'star', 'pentagon',
            'hexagon', 'octagon', 'star-triangle-up', 'star-triangle-down',
            'star-square', 'star-diamond', 'diamond-tall', 'diamond-wide',
            'hourglass', 'bowtie', 'circle-cross', 'circle-x', 'square-cross',
            'square-x', 'diamond-cross', 'diamond-x', 'cross-thin', 'x-thin',
            'asterisk', 'hash', 'y-up', 'y-down', 'y-left', 'y-right',
            'line-ew', 'line-ns', 'line-ne', 'line-nw'
        ]

        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def _generate_distinct_colors(self, n):
        """Generate n visually distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.15
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})')
        return colors

    def load_data(self):
        """Load benchmark data from JSON file"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        # Extract successful models
        successful_models = data['successful_models']

        rows = []
        for model in successful_models:
            row = {
                'model_name': model['model_name'],
                'file_extension': model['file_extension'],
                'file_size_mb': model['file_size_mb'],
                'resolution': model['resolution'],
                'map50_95': model['metrics']['map50_95'],
                'map50': model['metrics']['map50'],
                'precision': model['metrics']['precision'],
                'recall': model['metrics']['recall'],
                'inference_time_ms': model['timing']['avg_inference_time_ms'],
                'fps': model['timing']['fps'],
                'accuracy_speed_score': model['efficiency']['accuracy_speed_score']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        print(f"Loaded {len(df)} models for analysis")
        return df

    def setup_layout(self):
        """Setup the Dash app layout"""

        # Get unique values for filters
        formats = sorted(self.df['file_extension'].unique())
        resolutions = sorted(self.df['resolution'].unique())

        self.app.layout = html.Div([
            html.H1("Model Performance Interactive Analyzer",
                    style={'textAlign': 'center', 'marginBottom': 30}),

            # Control panel
            html.Div([
                html.Div([
                    html.Label("Select Metric for Y-axis:"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[
                            {'label': 'mAP@0.5:0.95', 'value': 'map50_95'},
                            {'label': 'mAP@0.5', 'value': 'map50'},
                            {'label': 'Precision', 'value': 'precision'},
                            {'label': 'Recall', 'value': 'recall'},
                            {'label': 'Accuracy×Speed Score', 'value': 'accuracy_speed_score'}
                        ],
                        value='map50_95'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),

                html.Div([
                    html.Label("Filter by Format:"),
                    dcc.Dropdown(
                        id='format-filter',
                        options=[{'label': 'All', 'value': 'all'}] +
                                [{'label': fmt.upper(), 'value': fmt} for fmt in formats],
                        value='all'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),

                html.Div([
                    html.Label("Filter by Resolution:"),
                    dcc.Dropdown(
                        id='resolution-filter',
                        options=[{'label': 'All', 'value': 'all'}] +
                                [{'label': f'{res}px', 'value': res} for res in resolutions],
                        value='all'
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ], style={'marginBottom': 30, 'padding': 20, 'backgroundColor': '#f0f0f0'}),

            # Main plot
            dcc.Graph(id='main-plot', style={'height': '600px'}),

            # Secondary plots
            html.Div([
                html.H3("Detailed Metrics Comparison", style={'textAlign': 'center'}),
                dcc.Graph(id='multi-plot', style={'height': '700px'})
            ], style={'marginTop': 30}),

            # Model details table
            html.Div([
                html.H3("Model Details", style={'textAlign': 'center'}),
                html.Div(id='model-table')
            ], style={'marginTop': 30})
        ])

    def setup_callbacks(self):
        """Setup interactive callbacks"""

        @self.app.callback(
            [Output('main-plot', 'figure'),
             Output('multi-plot', 'figure'),
             Output('model-table', 'children')],
            [Input('metric-dropdown', 'value'),
             Input('format-filter', 'value'),
             Input('resolution-filter', 'value')]
        )
        def update_plots(selected_metric, format_filter, resolution_filter):
            # Filter data
            filtered_df = self.df.copy()

            if format_filter != 'all':
                filtered_df = filtered_df[filtered_df['file_extension'] == format_filter]

            if resolution_filter != 'all':
                filtered_df = filtered_df[filtered_df['resolution'] == resolution_filter]

            # Main plot
            main_fig = self.create_main_plot(filtered_df, selected_metric)

            # Multi-plot
            multi_fig = self.create_multi_plot(filtered_df)

            # Model table
            table = self.create_model_table(filtered_df)

            return main_fig, multi_fig, table

    def create_main_plot(self, df, y_metric):
        """Create main performance plot"""

        fig = go.Figure()

        for idx, (_, row) in enumerate(df.iterrows()):
            color = self.colors[idx % len(self.colors)]
            marker = self.markers[idx % len(self.markers)]

            hover_text = (
                f"<b>{row['model_name']}</b><br>"
                f"Resolution: {row['resolution']}px<br>"
                f"Format: {row['file_extension']}<br>"
                f"Size: {row['file_size_mb']:.1f} MB<br>"
                f"FPS: {row['fps']:.1f}<br>"
                f"Inference: {row['inference_time_ms']:.1f} ms<br>"
                f"mAP@0.5:0.95: {row['map50_95']:.4f}<br>"
                f"Precision: {row['precision']:.4f}<br>"
                f"Recall: {row['recall']:.4f}"
            )

            fig.add_trace(
                go.Scatter(
                    x=[row['fps']],
                    y=[row[y_metric]],
                    mode='markers',
                    marker=dict(
                        symbol=marker,
                        size=max(10, row['file_size_mb'] * 2),
                        color=color,
                        line=dict(width=2, color='black'),
                        opacity=0.8
                    ),
                    name=row['model_name'],
                    hovertemplate=hover_text + "<extra></extra>",
                    showlegend=False
                )
            )

        # Add trend line if enough points
        if len(df) > 1:
            z = np.polyfit(df['fps'], df[y_metric], 1)
            x_trend = np.linspace(df['fps'].min(), df['fps'].max(), 100)
            y_trend = np.polyval(z, x_trend)

            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    line=dict(dash='dash', color='rgba(0,0,0,0.5)', width=2),
                    name='Trend Line',
                    hoverinfo='skip',
                    showlegend=False
                )
            )

        # Get metric display name
        metric_names = {
            'map50_95': 'mAP@0.5:0.95',
            'map50': 'mAP@0.5',
            'precision': 'Precision',
            'recall': 'Recall',
            'accuracy_speed_score': 'Accuracy×Speed Score'
        }

        fig.update_layout(
            title=f'Model Performance: FPS vs {metric_names[y_metric]}<br><sub>Hover for details • Marker size = File size • {len(df)} models shown</sub>',
            xaxis_title='Frames Per Second (FPS)',
            yaxis_title=metric_names[y_metric],
            hovermode='closest',
            height=600
        )

        return fig

    def create_multi_plot(self, df):
        """Create multi-metric comparison plot"""

        metrics = ['map50_95', 'map50', 'precision', 'recall']
        metric_names = ['mAP@0.5:0.95', 'mAP@0.5', 'Precision', 'Recall']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_names,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        for idx, (_, row) in enumerate(df.iterrows()):
            color = self.colors[idx % len(self.colors)]
            marker = self.markers[idx % len(self.markers)]

            hover_text = f"{row['model_name']}<br>{row['fps']:.1f} FPS"

            for i, metric in enumerate(metrics):
                row_idx = i // 2 + 1
                col_idx = i % 2 + 1

                fig.add_trace(
                    go.Scatter(
                        x=[row['fps']],
                        y=[row[metric]],
                        mode='markers',
                        marker=dict(
                            symbol=marker,
                            size=10,
                            color=color,
                            line=dict(width=1, color='black')
                        ),
                        name=row['model_name'],
                        hovertemplate=hover_text + f"<br>{metric_names[i]}: {row[metric]:.4f}<extra></extra>",
                        showlegend=False
                    ),
                    row=row_idx, col=col_idx
                )

        fig.update_layout(
            title='Multi-Metric Performance Analysis',
            height=700
        )

        # Update axes labels
        for i in range(2):
            for j in range(2):
                fig.update_xaxes(title_text="FPS", row=i + 1, col=j + 1)

        return fig

    def create_model_table(self, df):
        """Create model details table"""

        # Sort by mAP for display
        df_sorted = df.sort_values('map50_95', ascending=False)

        table_header = html.Thead([
            html.Tr([
                html.Th("Rank"),
                html.Th("Model Name"),
                html.Th("Resolution"),
                html.Th("Format"),
                html.Th("Size (MB)"),
                html.Th("mAP@0.5:0.95"),
                html.Th("FPS"),
                html.Th("Inference (ms)")
            ])
        ])

        table_rows = []
        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            table_rows.append(
                html.Tr([
                    html.Td(idx),
                    html.Td(row['model_name']),
                    html.Td(f"{row['resolution']}px"),
                    html.Td(row['file_extension']),
                    html.Td(f"{row['file_size_mb']:.1f}"),
                    html.Td(f"{row['map50_95']:.4f}"),
                    html.Td(f"{row['fps']:.1f}"),
                    html.Td(f"{row['inference_time_ms']:.1f}")
                ])
            )

        table_body = html.Tbody(table_rows)

        table = html.Table(
            [table_header, table_body],
            style={
                'width': '100%',
                'textAlign': 'center',
                'border': '1px solid black',
                'borderCollapse': 'collapse'
            }
        )

        return table

    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the Dash application"""
        print(f"\nStarting Interactive Model Analyzer")
        print(f"Loaded {len(self.df)} models")
        print(f"Open your browser and go to: http://{host}:{port}")
        print(f"Press Ctrl+C to stop the server\n")

        self.app.run_server(host=host, port=port, debug=debug)


class ModelBenchmarkAnalyzer:
    """Analyzes model performance metrics and inference speed"""

    def __init__(self,
                 models_folder: str,
                 dataset_yaml: str,
                 output_dir: str,
                 task: str = 'pose',
                 device: int = 0,
                 timing_runs: int = 100,
                 warmup_runs: int = 10,
                 recursive: bool = False):

        self.models_folder = models_folder
        self.dataset_yaml = dataset_yaml
        self.output_dir = output_dir
        self.task = task.lower()
        self.device = device
        self.timing_runs = timing_runs
        self.warmup_runs = warmup_runs
        self.recursive = recursive

        self.results = []
        self.timing_images = []

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate inputs
        self._validate_inputs()

        # Set up plotting style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'use') else 'default')
        sns.set_palette("husl")

    def _validate_inputs(self):
        """Validate all input paths exist"""
        if not os.path.exists(self.models_folder):
            raise FileNotFoundError(f"Models folder not found: {self.models_folder}")

        if not os.path.exists(self.dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")

        print("All input paths validated")
        print(f"Output directory: {self.output_dir}")
        print(f"Task type: {self.task}")
        print(f"Timing runs: {self.timing_runs} (warmup: {self.warmup_runs})")

    def _extract_resolution_from_name(self, model_name: str) -> int:
        """Extract resolution from model name (e.g., 'img832' -> 832)"""
        match = re.search(r'img(\d+)', model_name.lower())
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Could not extract resolution from '{model_name}', using default 640")
            return 640

    def find_model_files(self) -> List[Dict[str, Any]]:
        """Find all model files in the specified folder"""
        model_extensions = ['.pt', '.onnx', '.engine']
        model_files = []

        search_path = Path(self.models_folder)

        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in search_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in model_extensions:
                resolution = self._extract_resolution_from_name(file_path.stem)
                model_info = {
                    'name': file_path.stem,
                    'path': str(file_path),
                    'extension': file_path.suffix.lower(),
                    'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                    'resolution': resolution
                }
                model_files.append(model_info)
                print(f"Found model: {file_path.name} ({model_info['size_mb']} MB, {resolution}px)")

        print(f"\nFound {len(model_files)} model files to analyze")
        return sorted(model_files, key=lambda x: x['name'])

    def prepare_timing_images(self, num_images: int = 50) -> List[str]:
        """Prepare a set of images for timing measurements"""
        print(f"\nPreparing {num_images} images for timing measurements...")

        try:
            # Load dataset info to get image paths
            import yaml
            with open(self.dataset_yaml, 'r') as f:
                dataset_info = yaml.safe_load(f)

            # Get validation images path
            val_images_path = None
            if 'val' in dataset_info:
                val_images_path = os.path.join(dataset_info['path'], dataset_info['val'])
            elif 'path' in dataset_info:
                # Try common validation folder names
                base_path = Path(dataset_info['path'])
                for val_folder in ['val', 'valid', 'validation', 'test']:
                    potential_path = base_path / val_folder
                    if potential_path.exists():
                        val_images_path = str(potential_path)
                        break

            if not val_images_path or not os.path.exists(val_images_path):
                raise FileNotFoundError(f"Validation images path not found")

            # Find image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []

            for ext in image_extensions:
                image_files.extend(Path(val_images_path).glob(f"**/*{ext}"))
                image_files.extend(Path(val_images_path).glob(f"**/*{ext.upper()}"))

            if len(image_files) < num_images:
                print(f"Warning: Only found {len(image_files)} images, using all of them")
                num_images = len(image_files)

            # Randomly select images for timing
            selected_images = random.sample(image_files, num_images)
            self.timing_images = [str(img) for img in selected_images]

            print(f"Prepared {len(self.timing_images)} timing images")
            return self.timing_images

        except Exception as e:
            print(f"Warning: Could not prepare timing images: {e}")
            print("Will use synthetic data for timing")
            # Create synthetic timing data - this won't be accurate but allows testing
            self.timing_images = [f"synthetic_{i}.jpg" for i in range(num_images)]
            return self.timing_images

    def measure_inference_time(self, model, image_size: int = 640) -> float:
        """Measure average inference time for a model (excludes model loading time)"""
        print(f"  Measuring inference time...")

        times = []

        # Warmup runs
        print(f"    Warmup runs: {self.warmup_runs}")
        for _ in range(self.warmup_runs):
            if self.timing_images and os.path.exists(self.timing_images[0]):
                # Use real image
                img_path = random.choice(self.timing_images)
                try:
                    _ = model.predict(img_path, imgsz=image_size, device=self.device, verbose=False, batch=1)
                except:
                    # Fallback to synthetic data
                    dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                    _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)
            else:
                # Use synthetic data
                dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)

        # Timing runs
        print(f"    Timing runs: {self.timing_runs}")
        for i in range(self.timing_runs):
            if i % 20 == 0:
                print(f"    Progress: {i}/{self.timing_runs}")

            start_time = time.perf_counter()

            if self.timing_images and len(self.timing_images) > 0:
                # Use real image
                img_path = random.choice(self.timing_images)
                try:
                    if os.path.exists(img_path):
                        _ = model.predict(img_path, imgsz=image_size, device=self.device, verbose=False, batch=1)
                    else:
                        # Fallback
                        dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                        _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)
                except:
                    # Fallback to synthetic data
                    dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                    _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)
            else:
                # Use synthetic data
                dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False)

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"    Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        return avg_time, std_time, times

    def validate_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model and collect metrics + timing"""
        print(f"\n=== Analyzing {model_info['name']} ===")
        print(f"Path: {model_info['path']}")
        print(f"Size: {model_info['size_mb']} MB")
        print(f"Format: {model_info['extension']}")

        result = {
            'model_name': model_info['name'],
            'model_path': model_info['path'],
            'file_extension': model_info['extension'],
            'file_size_mb': model_info['size_mb'],
            'resolution': model_info['resolution'],
            'validation_success': False,
            'metrics': {},
            'timing': {}
        }

        try:
            # Load the model
            print("  Loading model...")
            model_load_start = time.time()
            model = YOLO(model_info['path'], task=self.task)
            model_load_time = time.time() - model_load_start
            print(f"  Model loaded in {model_load_time:.2f}s")

            # Run validation using the extracted resolution
            print("  Running validation...")
            image_size = model_info['resolution']
            print(f"  Using resolution for validation: {image_size}px")

            validation_start = time.time()
            validation_results = model.val(
                data=self.dataset_yaml,
                imgsz=image_size,  # Use extracted resolution
                batch=1,  # Use batch size 1 for streaming video use case
                device=self.device,
                verbose=False
            )
            validation_time = time.time() - validation_start

            # Extract metrics based on task type
            if validation_results is not None:
                metrics = {
                    'validation_time': validation_time,
                    'model_load_time': model_load_time,
                }

                # Extract metrics based on task type
                if self.task == 'pose' and hasattr(validation_results, 'pose') and validation_results.pose is not None:
                    metrics.update({
                        'map50_95': float(validation_results.pose.map),
                        'map50': float(validation_results.pose.map50),
                        'precision': float(validation_results.pose.mp),
                        'recall': float(validation_results.pose.mr),
                    })
                    print(f"  Using POSE metrics")
                elif hasattr(validation_results, 'box') and validation_results.box is not None:
                    metrics.update({
                        'map50_95': float(validation_results.box.map),
                        'map50': float(validation_results.box.map50),
                        'precision': float(validation_results.box.mp),
                        'recall': float(validation_results.box.mr),
                    })
                    print(f"  Using BBOX metrics")
                else:
                    # Fallback
                    print(f"  Warning: Could not extract metrics for task '{self.task}'")
                    metrics.update({
                        'map50_95': 0.0,
                        'map50': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                    })

                # Measure inference time using the resolution from model name
                image_size = model_info['resolution']
                print(f"  Using resolution: {image_size}px (extracted from model name)")

                avg_inference_time, std_inference_time, all_times = self.measure_inference_time(model, image_size)

                # Add timing information
                timing_info = {
                    'avg_inference_time_ms': avg_inference_time,
                    'std_inference_time_ms': std_inference_time,
                    'min_inference_time_ms': float(np.min(all_times)),
                    'max_inference_time_ms': float(np.max(all_times)),
                    'fps': 1000.0 / avg_inference_time,  # Convert ms to FPS
                    'image_size': image_size,
                    'timing_runs': self.timing_runs
                }

                result['metrics'] = metrics
                result['timing'] = timing_info
                result['validation_success'] = True

                print(f"  Validation completed in {validation_time:.2f}s")
                print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
                print(f"  mAP@0.5:     {metrics['map50']:.4f}")
                print(f"  Avg inference: {avg_inference_time:.2f} ms ({timing_info['fps']:.1f} FPS)")

            else:
                print("  Validation failed - no results returned")

        except Exception as e:
            print(f"  Analysis failed: {e}")
            result['error'] = str(e)

        finally:
            # Detailed cleanup for TensorRT models
            try:
                # Try to access TensorRT-specific components and delete in specified order
                if hasattr(model, 'predictor') and model.predictor is not None:
                    predictor = model.predictor

                    # 1. Context
                    if hasattr(predictor, 'context') and predictor.context is not None:
                        try:
                            del predictor.context
                            print(f"    Deleted TensorRT Context")
                        except:
                            pass

                    # 2. Engine
                    if hasattr(predictor, 'engine') and predictor.engine is not None:
                        try:
                            del predictor.engine
                            print(f"    Deleted TensorRT Engine")
                        except:
                            pass

                    # 3. Runtime
                    if hasattr(predictor, 'runtime') and predictor.runtime is not None:
                        try:
                            del predictor.runtime
                            print(f"    Deleted TensorRT Runtime")
                        except:
                            pass

                    # 4. Logger
                    if hasattr(predictor, 'logger') and predictor.logger is not None:
                        try:
                            del predictor.logger
                            print(f"    Deleted TensorRT Logger")
                        except:
                            pass

                    # Delete predictor
                    try:
                        del model.predictor
                        print(f"    Deleted Predictor")
                    except:
                        pass

                # Delete the model itself
                try:
                    del model
                    print(f"    Deleted Model")
                except:
                    pass

            except Exception as cleanup_error:
                print(f"    Warning during TensorRT cleanup: {cleanup_error}")

            # Force garbage collection
            gc.collect()

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                print(f"  GPU memory cleaned and synchronized")

        return result

    def analyze_all_models(self) -> List[Dict[str, Any]]:
        """Analyze all models in the folder"""
        model_files = self.find_model_files()

        if not model_files:
            print("No model files found to analyze!")
            return []

        # Prepare timing images
        self.prepare_timing_images()

        print(f"\n{'=' * 60}")
        print(f"ANALYZING {len(model_files)} MODELS")
        print(f"{'=' * 60}")

        for i, model_info in enumerate(model_files, 1):
            print(f"\n[{i}/{len(model_files)}] Processing {model_info['name']}")

            result = self.validate_model(model_info)
            self.results.append(result)

        return self.results

    def generate_enhanced_visualizations(self):
        """Generate enhanced visualizations with support for 100+ models"""
        successful_results = [r for r in self.results if r['validation_success']]

        if not successful_results:
            print("No successful results to visualize!")
            return

        # Create DataFrame
        plot_data = []
        for result in successful_results:
            metrics = result['metrics']
            timing = result['timing']

            plot_data.append({
                'model_name': result['model_name'],
                'file_extension': result['file_extension'],
                'file_size_mb': result['file_size_mb'],
                'resolution': result['resolution'],
                'inference_time_ms': timing['avg_inference_time_ms'],
                'fps': timing['fps'],
                'map50_95': metrics['map50_95'],
                'map50': metrics['map50'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })

        df = pd.DataFrame(plot_data)

        # Create enhanced visualizer
        visualizer = EnhancedModelVisualizer()

        # Generate all visualization types
        print("Generating enhanced visualizations...")

        # 1. Interactive multi-plot analysis
        visualizer.create_interactive_plots(df, self.output_dir, self.task)

        # 2. Main performance plot (interactive)
        visualizer.create_main_performance_plot(df, self.output_dir)

        # 3. Enhanced static plots with reference guide
        visualizer.create_enhanced_static_plots(df, self.output_dir, self.task)

        print(f"\nEnhanced visualizations complete!")
        print(f"  Interactive plots: {self.output_dir}/interactive_performance_analysis.html")
        print(f"  Main plot: {self.output_dir}/main_performance_plot.html")
        print(f"  Reference guide: {self.output_dir}/model_reference_guide.png")
        print(f"  Static plots: {self.output_dir}/enhanced_static_performance.png")
        print(
            f"\nSupports 1500+ unique color-shape combinations ({len(visualizer.colors)} colors × {len(visualizer.markers)} shapes)")
        print(f"  Best for many models: Open the .html files in your browser for interactive exploration")

    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            print("No results to generate report from!")
            return {}

        successful_results = [r for r in self.results if r['validation_success']]
        failed_results = [r for r in self.results if not r['validation_success']]

        # Sort by mAP for ranking
        successful_results.sort(key=lambda x: x['metrics'].get('map50_95', 0), reverse=True)

        # Calculate efficiency metrics
        for result in successful_results:
            if result['validation_success']:
                metrics = result['metrics']
                timing = result['timing']

                # Efficiency metrics
                map_per_mb = metrics['map50_95'] / max(result['file_size_mb'], 0.1)
                map_per_ms = metrics['map50_95'] / max(timing['avg_inference_time_ms'], 0.1)

                result['efficiency'] = {
                    'map_per_mb': map_per_mb,
                    'map_per_ms': map_per_ms,
                    'accuracy_speed_score': metrics['map50_95'] * timing['fps'] / 100  # Normalized score
                }

        # Create comprehensive report
        report = {
            'analysis_info': {
                'models_folder': self.models_folder,
                'dataset_yaml': self.dataset_yaml,
                'task_type': self.task,
                'timing_runs': self.timing_runs,
                'warmup_runs': self.warmup_runs,
                'device': self.device,
                'total_models': len(self.results),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(failed_results)
            },
            'successful_models': successful_results,
            'failed_models': failed_results,
            'rankings': {
                'best_accuracy': successful_results[:5] if successful_results else [],
                'fastest_models': sorted(successful_results,
                                         key=lambda x: x['timing']['avg_inference_time_ms'])[:5],
                'most_efficient': sorted(successful_results,
                                         key=lambda x: x['efficiency']['accuracy_speed_score'],
                                         reverse=True)[:5] if successful_results else []
            }
        }

        # Save report
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'benchmark_report.json')

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to: {output_file}")

        # Save CSV summary
        self.save_csv_summary(successful_results)

        return report

    def save_csv_summary(self, successful_results: List[Dict]):
        """Save CSV summary of all results"""
        csv_file = os.path.join(self.output_dir, 'benchmark_summary.csv')

        try:
            import csv

            with open(csv_file, 'w', newline='') as f:
                fieldnames = [
                    'model_name', 'file_extension', 'file_size_mb', 'resolution',
                    'map50_95', 'map50', 'precision', 'recall',
                    'avg_inference_time_ms', 'fps', 'accuracy_speed_score'
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in successful_results:
                    row = {
                        'model_name': result['model_name'],
                        'file_extension': result['file_extension'],
                        'file_size_mb': result['file_size_mb'],
                        'resolution': result['resolution'],
                        'map50_95': result['metrics']['map50_95'],
                        'map50': result['metrics']['map50'],
                        'precision': result['metrics']['precision'],
                        'recall': result['metrics']['recall'],
                        'avg_inference_time_ms': result['timing']['avg_inference_time_ms'],
                        'fps': result['timing']['fps'],
                        'accuracy_speed_score': result['efficiency']['accuracy_speed_score']
                    }
                    writer.writerow(row)

            print(f"CSV summary saved to: {csv_file}")

        except Exception as e:
            print(f"Warning: Could not save CSV summary: {e}")

    def print_summary(self, report: Dict[str, Any]):
        """Print summary to console"""
        print(f"\n{'=' * 80}")
        print("MODEL BENCHMARK ANALYSIS SUMMARY")
        print(f"{'=' * 80}")

        info = report['analysis_info']
        print(f"Task type: {info['task_type']}")
        print(f"Models analyzed: {info['total_models']}")
        print(f"Successful: {info['successful_analyses']}")
        print(f"Failed: {info['failed_analyses']}")
        print(f"Timing runs per model: {info['timing_runs']}")

        if report['rankings']['best_accuracy']:
            print(f"\nTOP 5 BY ACCURACY (mAP@0.5:0.95):")
            print("-" * 115)
            print(
                f"{'Rank':<4} {'Model':<25} {'Resolution':<10} {'Format':<8} {'mAP@0.5:0.95':<12} {'Time(ms)':<10} {'FPS':<8} {'Size(MB)':<10}")
            print("-" * 115)

            for i, model in enumerate(report['rankings']['best_accuracy'], 1):
                print(f"{i:<4} {model['model_name']:<25} {model['resolution']:<10} {model['file_extension']:<8} "
                      f"{model['metrics']['map50_95']:.4f}      "
                      f"{model['timing']['avg_inference_time_ms']:.1f}      "
                      f"{model['timing']['fps']:.1f}   "
                      f"{model['file_size_mb']:.1f}")

        if report['rankings']['fastest_models']:
            print(f"\nTOP 5 FASTEST MODELS:")
            print("-" * 115)
            print(
                f"{'Rank':<4} {'Model':<25} {'Resolution':<10} {'Format':<8} {'Time(ms)':<10} {'FPS':<8} {'mAP@0.5:0.95':<12} {'Size(MB)':<10}")
            print("-" * 115)

            for i, model in enumerate(report['rankings']['fastest_models'], 1):
                print(f"{i:<4} {model['model_name']:<25} {model['resolution']:<10} {model['file_extension']:<8} "
                      f"{model['timing']['avg_inference_time_ms']:.1f}      "
                      f"{model['timing']['fps']:.1f}   "
                      f"{model['metrics']['map50_95']:.4f}      "
                      f"{model['file_size_mb']:.1f}")

        if report['rankings']['most_efficient']:
            print(f"\nTOP 5 MOST EFFICIENT (Accuracy×Speed Score):")
            print("-" * 125)
            print(
                f"{'Rank':<4} {'Model':<25} {'Resolution':<10} {'Format':<8} {'Score':<8} {'mAP@0.5:0.95':<12} {'Time(ms)':<10} {'FPS':<8}")
            print("-" * 125)

            for i, model in enumerate(report['rankings']['most_efficient'], 1):
                print(f"{i:<4} {model['model_name']:<25} {model['resolution']:<10} {model['file_extension']:<8} "
                      f"{model['efficiency']['accuracy_speed_score']:.2f}   "
                      f"{model['metrics']['map50_95']:.4f}      "
                      f"{model['timing']['avg_inference_time_ms']:.1f}      "
                      f"{model['timing']['fps']:.1f}")

    def launch_interactive_analyzer(self, host='127.0.0.1', port=8050):
        """Launch the interactive offline analyzer"""
        benchmark_json = os.path.join(self.output_dir, 'benchmark_report.json')

        if not os.path.exists(benchmark_json):
            print(f"Benchmark report not found at {benchmark_json}")
            print("Run the analysis first to generate the report.")
            return

        try:
            analyzer = OfflineModelAnalyzer(benchmark_json)
            analyzer.run(host=host, port=port)
        except Exception as e:
            print(f"Error launching interactive analyzer: {e}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Model Performance and Speed Benchmark Analyzer')

    parser.add_argument('--models-folder', required=True,
                        help='Folder containing model files to analyze')
    parser.add_argument('--dataset-yaml', required=True,
                        help='Path to validation dataset YAML file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for reports and visualizations')
    parser.add_argument('--task', choices=['pose', 'detect', 'segment'], default='pose',
                        help='Model task type')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--timing-runs', type=int, default=100,
                        help='Number of timing runs per model')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Number of warmup runs before timing')
    parser.add_argument('--recursive', action='store_true',
                        help='Search for models recursively in subfolders')
    parser.add_argument('--output-json',
                        help='Custom output path for JSON report')
    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive analyzer after benchmarking')
    parser.add_argument('--interactive-only', action='store_true',
                        help='Only launch interactive analyzer (skip benchmarking)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Host for interactive analyzer')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port for interactive analyzer')

    args = parser.parse_args()

    if args.interactive_only:
        # Only launch interactive analyzer
        benchmark_json = os.path.join(args.output_dir, 'benchmark_report.json')
        if not os.path.exists(benchmark_json):
            print(f"Benchmark report not found at {benchmark_json}")
            print("Run the analysis first without --interactive-only to generate the report.")
            sys.exit(1)

        try:
            analyzer = OfflineModelAnalyzer(benchmark_json)
            analyzer.run(host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\nAnalyzer stopped by user")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    print("Starting Enhanced Model Benchmark Analysis")
    print(f"Models folder: {args.models_folder}")
    print(f"Dataset: {args.dataset_yaml}")
    print(f"Task: {args.task}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timing runs: {args.timing_runs}")

    try:
        # Create analyzer
        analyzer = ModelBenchmarkAnalyzer(
            models_folder=args.models_folder,
            dataset_yaml=args.dataset_yaml,
            output_dir=args.output_dir,
            task=args.task,
            device=args.device,
            timing_runs=args.timing_runs,
            warmup_runs=args.warmup_runs,
            recursive=args.recursive
        )

        # Analyze all models
        results = analyzer.analyze_all_models()

        if not results:
            print("No models were analyzed!")
            sys.exit(1)

        # Generate enhanced visualizations
        analyzer.generate_enhanced_visualizations()

        # Generate and save report
        report = analyzer.generate_report(args.output_json)

        # Print summary
        analyzer.print_summary(report)

        print(f"\nAnalysis completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Interactive visualizations: *.html files")
        print(f"Static visualizations: *.png files")
        print(f"Reports: benchmark_report.json, benchmark_summary.csv")

        # Launch interactive analyzer if requested
        if args.interactive:
            print(f"\nLaunching interactive analyzer...")
            analyzer.launch_interactive_analyzer(host=args.host, port=args.port)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Usage examples:
"""
INSTALLATION:
pip install ultralytics matplotlib seaborn pandas pillow torch plotly dash

BASIC USAGE:
python enhanced_benchmark_analyzer.py \
    --models-folder ./trained_models \
    --dataset-yaml ./validation_dataset.yaml \
    --output-dir ./benchmark_results \
    --task pose

ENHANCED USAGE WITH INTERACTIVE ANALYZER:
python enhanced_benchmark_analyzer.py \
    --models-folder ./models \
    --dataset-yaml ./dataset.yaml \
    --output-dir ./results \
    --task detect \
    --interactive

LAUNCH ONLY INTERACTIVE ANALYZER (after running analysis):
python enhanced_benchmark_analyzer.py \
    --output-dir ./results \
    --interactive-only \
    --host 0.0.0.0 \
    --port 8080

FEATURES:
- Interactive HTML plots with hover details
- Static plots with 1500+ unique color-shape combinations
- Model reference guides
- Offline interactive web application with filtering
- Support for 100+ models with unique visual identification
"""