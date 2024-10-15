from flask import Flask, render_template, request, send_from_directory, jsonify, url_for
import os
from moviepy.editor import VideoFileClip
import torch
import numpy as np
import consts
import training
from dataset_creation import extract_frames
from model import EfficientNetLSTMModel # noqa: F401

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['PROCESSING_FOLDER'] = 'processing'
os.makedirs(app.config['PROCESSING_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)

        output_filename = 'output_' + file.filename
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        try:
            success = process_video(input_path, output_path, file)
            if not success:
                return jsonify({'output_file': None, 'error': 'No output file created'}), 200
            return jsonify({
                'output_file': url_for('download_file', filename=output_filename),
                'download_link': url_for('download_file', filename=output_filename)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500


def max_sequence_of_ones(predictions):
    max_length = 0
    current_length = 0
    start_index = -1
    max_start_index = -1
    for i in range(len(predictions[0])):
        if predictions[0][i] == 1:
            if current_length == 0:
                start_index = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start_index = start_index
            current_length = 0

    # Check at the end of the loop in case the longest sequence ends at the last element
    if current_length > max_length:
        max_length = current_length
        max_start_index = start_index
    return max_start_index, max_length


def process_video(input_path, output_path, file):
    processing_filename = 'processing_' + file.filename
    processing_path = os.path.join(app.config['PROCESSING_FOLDER'], processing_filename)
    clip = VideoFileClip(input_path)
    clip.write_videofile(processing_path, fps=consts.FPS, audio=False)
    clip.close()
    frames = extract_frames(str(processing_path))
    frames = np.array(frames)
    model = torch.load(consts.PRODUCTION_MODEL_PATH, map_location=consts.DEVICE)
    predictions = training.evaluate(model, [frames], labeled=False)
    max_start_index, max_length = max_sequence_of_ones(predictions)
    if max_length == 0:
        return False
    start = max_start_index // consts.FPS
    end = ((max_start_index + max_length) // 2) + 1
    clip = VideoFileClip(input_path)
    video = clip.subclip(start, end)
    video.write_videofile(output_path)
    clip.close()
    return True


if __name__ == '__main__':
    app.run()
