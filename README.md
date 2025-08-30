# Subgen: Automatic Subtitle Generator
Subgen is a simple yet powerful command-line tool that automatically generates SRT subtitles for video and audio files. It is built on top of the highly efficient faster-whisper library, which provides a fast and accurate speech-to-text transcription engine.

This project is 100% computer generated. If you find any issues, please open an issue or submit a pull request.

## Features
🗣️ Automatic Subtitle Generation: Transcribes audio from video and audio files to create accurate .srt subtitle files.

📁 Single File or Directory Processing: Handles both individual video files and entire directories of media files.

⏩ Resumable Batch Processing: Use the --skip argument to continue processing a directory from a specific file, making it easy to recover from interruptions.

⬇️ Automatic Model Download: The required faster-whisper model is automatically downloaded from the Hugging Face Hub if it's not found on your system.

🔧 Configurable: Provides command-line options to customize the model, device, and other transcription parameters.

## Prerequisites
- Python 3.8 or higher.

- ffmpeg: The ffmpeg tool must be installed on your system and available in your system's PATH. This is required for faster-whisper to handle various video and audio formats.

## Installation
Clone the repository:

```Bash
git clone https://github.com/YasserHawass/subtitles-generator.git
cd subtitles-generator
```
Create a virtual environment (recommended):
```Bash
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
# or
conda create --name subgen
conda activate subgen
```
Install dependencies:
```Bash
pip install -r requirements.txt
# or
conda install -c nvidia/label/cuda-12.4.0 cudnn
```

## Usage
The tool's main entry point is subgen.py.
Single File
To generate subtitles for a single video file, simply provide the path to the file.

```Bash
python subgen.py "/path/to/your/video.mp4"
```
The output video.srt will be created in the same directory as the video.

Directory
To process all supported media files within a directory, use the --dir flag.

```Bash
python subgen.py "/path/to/your/videos" --dir

Options
The following options are available to customize the behavior of the script:

Flag	Description	Default
--dir	Process the input path as a directory.	False
--recursive	Recurse into subdirectories when used with --dir.	False
--skip N	Skip the first N videos in a directory.	0
-o, --output-dir PATH	Write SRT files to a specific directory.	Alongside each video
--overwrite	Overwrite existing .srt files.	False
--model NAME	Specify the model to download and use (e.g., medium, large-v3).	base
--language CODE	Force transcription to a specific language (e.g., en, ar).	Auto-detect
--no-vad	Disable voice activity detection (VAD).	False
-v, --verbose	Enable verbose logging.	False
```

Export to Sheets
Example: Process a directory, skipping the first 5 videos and using a larger model.

```Bash
python subgen.py "/path/to/your/videos" --dir --skip 5 --model large-v3
```

## Contributing
We welcome contributions! Please feel free to open an issue or submit a pull request if you find a bug or have an idea for a new feature.

## License
This project is licensed under the MIT License.