# JPEG_Codec

# JPEG Codec - Readme

## Overview
This Python application provides a graphical user interface (GUI) for compressing and decompressing images using a simplified JPEG-like algorithm. The tool implements key components of JPEG compression including Discrete Cosine Transform (DCT), quantization, and zigzag scanning.

## Features

- **Image Compression**:
  - Convert images to JPEG-compressed format
  - Adjustable compression quality (1-100)
  - Option to process as grayscale
  - Saves compressed data in NPZ format

- **Image Decompression**:
  - Reconstruct images from compressed NPZ files
  - Preview functionality for compressed files
  - Save decompressed images in JPEG or PNG format

- **GUI Interface**:
  - Intuitive controls for compression/decompression
  - Image preview before and after processing
  - File size comparison information

## Requirements

- Python 3.x
- Required packages:
  - OpenCV (`cv2`)
  - NumPy
  - Pillow (PIL)
  - Tkinter (usually included with Python)

Install dependencies with:
```
pip install opencv-python numpy pillow
```

## Usage

1. **Launch the application**:
   ```
   python JPEG_Codec.py
   ```

2. **Compressing an image**:
   - Click "Select Input File" and choose an image (JPG, JPEG, PNG)
   - Adjust quality slider (1-100)
   - Check "Grayscale" option if desired
   - Click "Select Output Path" to choose where to save
   - Click "Compress Image"

3. **Decompressing an image**:
   - Click "Select Input File" and choose a .npz compressed file
   - Click "Select Output Path" to choose where to save
   - Click "Decompress Image"
   - Optionally use "Save Preview" to save the displayed preview

## File Formats

- **Input**: JPG, JPEG, PNG (for compression) or NPZ (for decompression)
- **Output**: 
  - Compressed data: NPZ format
  - Decompressed images: JPG, JPEG, or PNG

## Technical Details

The implementation includes:
- 2D Discrete Cosine Transform (DCT) using OpenCV
- Quantization with adjustable quality tables
- Zigzag scanning for efficient encoding
- Block processing with 8x8 pixel blocks
- Color space conversion (RGB/BGR/YUV)

## Notes

- Higher quality values result in larger files but better image quality
- Grayscale compression typically produces smaller files
- The NPZ format preserves all compression data for accurate reconstruction

## License

This project is open-source and available for free use. Modify and distribute as needed.

---

For any issues or questions, please refer to the code comments or open an issue in the project repository.