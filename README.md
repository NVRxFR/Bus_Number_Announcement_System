# Blind-Friendly Bus Identification System

## Overview
This system helps visually impaired individuals identify approaching buses through real-time video capture and audio announcements. By leveraging Intel's OpenVINO, oneDNN, and MKL, the system ensures high performance and efficiency in detecting bus numbers.

## Features
* Real-time bus number detection
* Audio announcements of approaching buses
* Optimized using Intel OpenVINO, oneDNN, and MKL for high performance

## Setup
1. **Install Python**: Ensure you have Python (version 3.8 or above) installed on your machine.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv bus_detection_env
   ```
3. **Activate the Virtual Environment**:
   * On Windows:
     ```bash
     bus_detection_env\Scripts\activate
     ```
   * On macOS and Linux:
     ```bash
     source bus_detection_env/bin/activate
     ```
4. **Install Intel OpenVINO**: Follow the official OpenVINO installation guide for your operating system.
5. **Install Required Dependencies**: Create a `requirements.txt` file with the necessary packages:
   ```text
   opencv-python
   easyocr
   gTTS
   torch
   torchvision
   oneDNN
   numpy
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main application:
```bash
python src/main.py
```

## Project Structure
```
busannouncer/
├── mods/
│  ├── tests/
│   ├── video_capture.py    # Video capture functionality
│   ├── text_detection.py   # Text detection using EasyOCR
│   ├── text_to_speech.py   # Text-to-speech functionality
│   └── main.py             # Main application file
                  # Test files
├── docs/                   # Documentation
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Testing
Run tests using:
```bash
pytest tests/
```

## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Notes
* **Intel MKL**: Ensure MKL is installed as part of the OpenVINO installation.
* **oneDNN**: The oneDNN library is typically included with Intel's oneAPI toolkit, but make sure to verify its installation.
