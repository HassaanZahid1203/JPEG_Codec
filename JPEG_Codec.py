import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# --- JPEG Coder ---
def dct2(block):
    """Apply 2D Discrete Cosine Transform to an 8x8 block."""
    return cv2.dct(block.astype(np.float32))

def quantize(block, q_table):
    """Quantize the DCT coefficients using a quantization table."""
    return np.round(block / q_table).astype(np.int32)

def zigzag_scan(block):
    """Perform zigzag scanning on an 8x8 block to produce a 1D array."""
    zigzag_order = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    return [block[i][j] for i, j in zigzag_order]

def encode_image(img, q_table, is_grayscale=False):
    """Encode an image using JPEG compression, returning quantized coefficients."""
    h, w = img.shape[:2]
    channels = 1 if is_grayscale else 3
    encoded_data = []

    # Process each 8x8 block
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            for c in range(channels):
                block = img[i:i+8, j:j+8, c] if not is_grayscale else img[i:i+8, j:j+8]
                if block.shape != (8, 8):
                    # Pad block if necessary
                    block = np.pad(block, ((0, 8 - block.shape[0]), (0, 8 - block.shape[1])), mode='constant')
               
                # Subtract 128 to center pixel values
                block = block.astype(np.float32) - 128
                # Apply DCT
                dct_block = dct2(block)
                # Quantize
                quantized = quantize(dct_block, q_table)
                # Zigzag scan
                zigzagged = zigzag_scan(quantized)
                encoded_data.append(zigzagged)
   
    return encoded_data, h, w

# --- JPEG Decoder ---
def inverse_zigzag(data, block_size=64):
    """Convert 1D zigzag array back to 8x8 block."""
    zigzag_order = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    block = np.zeros((8, 8), dtype=np.float32)
    for idx, (i, j) in enumerate(zigzag_order):
        if idx < len(data):
            block[i, j] = data[idx]
    return block

def dequantize(block, q_table):
    """Dequantize the block using the quantization table."""
    return block * q_table

def idct2(block):
    """Apply Inverse 2D Discrete Cosine Transform to an 8x8 block."""
    return cv2.idct(block)

def decode_image(encoded_data, h, w, q_table, is_grayscale=False):
    """Decode the compressed data back to an image."""
    channels = 1 if is_grayscale else 3
    decoded_img = np.zeros((h, w, channels) if not is_grayscale else (h, w), dtype=np.float32)
    block_idx = 0

    # Process each 8x8 block
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            for c in range(channels):
                if block_idx >= len(encoded_data):
                    break  # Prevent index out of range
                # Extract 64 coefficients for the block
                block_data = encoded_data[block_idx]
                block_idx += 1
                # Inverse zigzag
                block = inverse_zigzag(block_data)
                # Dequantize
                dequantized = dequantize(block, q_table)
                # Apply IDCT
                decoded_block = idct2(dequantized)
                # Add 128 to shift pixel values back
                decoded_block += 128
                # Place block in image
                h_slice = min(8, h - i)
                w_slice = min(8, w - j)
                if is_grayscale:
                    decoded_img[i:i+h_slice, j:j+w_slice] = decoded_block[:h_slice, :w_slice]
                else:
                    decoded_img[i:i+h_slice, j:j+w_slice, c] = decoded_block[:h_slice, :w_slice]
   
    return np.clip(decoded_img, 0, 255).astype(np.uint8)

# --- Quantization Table ---
def get_quantization_table(quality=50):
    """Generate quantization table based on quality (1-100)."""
    # Standard JPEG quantization table for luminance
    q_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    # Scale based on quality (more aggressive compression for lower quality)
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q_table = np.clip((q_table * scale + 50) / 100, 1, 255)
    return q_table

# --- GUI ---
class JPEGCompressorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG Compressor and Decompressor")
        self.root.geometry("800x600")

        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.compressed_data_path = tk.StringVar()
        self.quality = tk.IntVar(value=50)
        self.is_grayscale = tk.BooleanVar(value=False)
        self.file_size_info = tk.StringVar(value="File sizes: Not loaded")
        self.image = None
        self.is_npz_input = tk.BooleanVar(value=False)
        self.preview_decoded_img = None  # Store decompressed image for .npz preview
        self.preview_is_grayscale = False  # Store grayscale flag for .npz preview

        # GUI Elements
        tk.Label(root, text="JPEG Compressor and Decompressor", font=("Arial", 16)).pack(pady=10)

        # Input File
        tk.Button(root, text="Select Input File", command=self.load_file).pack(pady=5)
        tk.Label(root, textvariable=self.input_path).pack()

        # Quality Slider
        tk.Label(root, text="Quality (1-100):").pack()
        self.quality_scale = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.quality)
        self.quality_scale.pack()

        # Grayscale Checkbox
        self.grayscale_check = tk.Checkbutton(root, text="Grayscale", variable=self.is_grayscale)
        self.grayscale_check.pack()

        # Compress Button
        self.compress_button = tk.Button(root, text="Compress Image", command=self.compress_image, state=tk.DISABLED)
        self.compress_button.pack(pady=10)

        # Decompress Button
        self.decompress_button = tk.Button(root, text="Decompress Image", command=self.decompress_image, state=tk.DISABLED)
        self.decompress_button.pack(pady=5)

        # Save Preview Button (for .npz previews)
        self.save_preview_button = tk.Button(root, text="Save Preview", command=self.save_preview, state=tk.DISABLED)
        self.save_preview_button.pack(pady=5)

        # Output Path
        tk.Button(root, text="Select Output Path", command=self.select_output).pack(pady=5)
        tk.Label(root, textvariable=self.output_path).pack()

        # File Size Display
        tk.Label(root, textvariable=self.file_size_info).pack(pady=5)

        # Display Area
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def load_file(self):
        """Load and display the input file (image or .npz), and update file size info."""
        path = filedialog.askopenfilename(filetypes=[
            ("All Supported Files", "*.jpg *.jpeg *.png *.npz"),
            ("Image Files", "*.jpg *.jpeg *.png"),
            ("Compressed Data", "*.npz")
        ])
        if not path:
            return

        self.input_path.set(path)
        self.is_npz_input.set(path.lower().endswith('.npz'))
        self.preview_decoded_img = None  # Reset preview image
        self.preview_is_grayscale = False

        if self.is_npz_input.get():
            # Handle .npz file (decompression preview)
            try:
                data = np.load(path, allow_pickle=True)
                encoded_data = data['encoded_data'].tolist()
                h = data['h']
                w = data['w']
                q_table = data['q_table']
                self.preview_is_grayscale = data['is_grayscale']
               
                # Decode for preview
                self.preview_decoded_img = decode_image(encoded_data, h, w, q_table, self.preview_is_grayscale)
               
                # Convert to PIL Image for display
                if self.preview_is_grayscale:
                    pil_img = Image.fromarray(self.preview_decoded_img)
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(self.preview_decoded_img, cv2.COLOR_RGB2BGR))
                pil_img.thumbnail((300, 300))
                self.image = ImageTk.PhotoImage(pil_img)
                self.image_label.config(image=self.image)
               
                # Update file size info
                input_size = os.path.getsize(path) / 1024  # Size in KB
                self.file_size_info.set(f"Compressed data size: {input_size:.2f} KB | Output file size: Not saved")
               
                # Enable decompress and save preview buttons, disable compress button and options
                self.compress_button.config(state=tk.DISABLED)
                self.decompress_button.config(state=tk.NORMAL)
                self.save_preview_button.config(state=tk.NORMAL)
                self.quality_scale.config(state=tk.DISABLED)
                self.grayscale_check.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load .npz file: {str(e)}")
                self.input_path.set("")
                self.image_label.config(image="")
                self.file_size_info.set("File sizes: Not loaded")
                self.save_preview_button.config(state=tk.DISABLED)
        else:
            # Handle image file
            try:
                img = Image.open(path)
                img.thumbnail((300, 300))  # Resize for display
                self.image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.image)
               
                # Update file size info
                input_size = os.path.getsize(path) / 1024  # Size in KB
                self.file_size_info.set(f"Input file size: {input_size:.2f} KB | Output file size: Not compressed")
               
                # Enable compress button, disable decompress and save preview buttons
                self.compress_button.config(state=tk.NORMAL)
                self.decompress_button.config(state=tk.DISABLED)
                self.save_preview_button.config(state=tk.DISABLED)
                self.quality_scale.config(state=tk.NORMAL)
                self.grayscale_check.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.input_path.set("")
                self.image_label.config(image="")
                self.file_size_info.set("File sizes: Not loaded")
                self.save_preview_button.config(state=tk.DISABLED)

    def select_output(self):
        """Select output file path with multiple extension options."""
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.output_path.set(path)

    def compress_image(self):
        """Compress the image and save compressed data."""
        if not self.input_path.get() or not self.output_path.get():
            messagebox.showerror("Error", "Please select input and output paths.")
            return

        # Read image
        img = cv2.imread(self.input_path.get())
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        # Convert to grayscale if selected, else convert BGR to RGB
        if self.is_grayscale.get():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get quantization table
        q_table = get_quantization_table(self.quality.get())

        # Encode image
        encoded_data, h, w = encode_image(img, q_table, self.is_grayscale.get())

        # Save compressed data to a .npz file
        compressed_data_path = self.output_path.get().rsplit('.', 1)[0] + '_compressed.npz'
        np.savez(compressed_data_path,
                 encoded_data=np.array(encoded_data, dtype=object),
                 h=h,
                 w=w,
                 q_table=q_table,
                 is_grayscale=self.is_grayscale.get())
        self.compressed_data_path.set(compressed_data_path)

        # Decode image for saving and display
        try:
            decoded_img = decode_image(encoded_data, h, w, q_table, self.is_grayscale.get())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to decode image: {str(e)}")
            return

        # Save output image
        quality = self.quality.get()
        output_ext = os.path.splitext(self.output_path.get())[1].lower()
        try:
            if output_ext in ('.jpg', '.jpeg'):
                if self.is_grayscale.get():
                    cv2.imwrite(self.output_path.get(), decoded_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    cv2.imwrite(self.output_path.get(), cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            elif output_ext == '.png':
                if self.is_grayscale.get():
                    cv2.imwrite(self.output_path.get(), decoded_img)
                else:
                    cv2.imwrite(self.output_path.get(), cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR))
            else:
                messagebox.showerror("Error", "Unsupported output format. Use .jpg, .jpeg, or .png.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            return

        # Calculate and display file sizes
        input_size = os.path.getsize(self.input_path.get()) / 1024  # Size in KB
        output_size = os.path.getsize(self.output_path.get()) / 1024  # Size in KB
        compressed_data_size = os.path.getsize(compressed_data_path) / 1024  # Size in KB
        self.file_size_info.set(f"Input file size: {input_size:.2f} KB | Output file size: {output_size:.2f} KB | Compressed data size: {compressed_data_size:.2f} KB")

        # Show success message
        messagebox.showinfo("Success", f"Image compressed successfully! Compressed data saved to {compressed_data_path}")

        # Display compressed image
        try:
            compressed_img = Image.open(self.output_path.get())
            compressed_img.thumbnail((300, 300))
            self.image = ImageTk.PhotoImage(compressed_img)
            self.image_label.config(image=self.image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display compressed image: {str(e)}")

    def decompress_image(self):
        """Decompress a .npz file and save the image."""
        if not self.input_path.get() or not self.output_path.get():
            messagebox.showerror("Error", "Please select input and output paths.")
            return

        # Load compressed data
        try:
            data = np.load(self.input_path.get(), allow_pickle=True)
            encoded_data = data['encoded_data'].tolist()
            h = data['h']
            w = data['w']
            q_table = data['q_table']
            is_grayscale = data['is_grayscale']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load compressed data: {str(e)}")
            return

        # Decode image
        try:
            decoded_img = decode_image(encoded_data, h, w, q_table, is_grayscale)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to decode image: {str(e)}")
            return

        # Save decompressed image
        quality = self.quality.get()
        output_ext = os.path.splitext(self.output_path.get())[1].lower()
        try:
            if output_ext in ('.jpg', '.jpeg'):
                if is_grayscale:
                    cv2.imwrite(self.output_path.get(), decoded_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    cv2.imwrite(self.output_path.get(), cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            elif output_ext == '.png':
                if is_grayscale:
                    cv2.imwrite(self.output_path.get(), decoded_img)
                else:
                    cv2.imwrite(self.output_path.get(), cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR))
            else:
                messagebox.showerror("Error", "Unsupported output format. Use .jpg, .jpeg, or .png.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save decompressed image: {str(e)}")
            return

        # Update file size info
        compressed_data_size = os.path.getsize(self.input_path.get()) / 1024  # Size in KB
        output_size = os.path.getsize(self.output_path.get()) / 1024  # Size in KB
        self.file_size_info.set(f"Compressed data size: {compressed_data_size:.2f} KB | Decompressed file size: {output_size:.2f} KB")

        # Show success message
        messagebox.showinfo("Success", "Image decompressed successfully!")

        # Display decompressed image
        try:
            decompressed_img = Image.open(self.output_path.get())
            decompressed_img.thumbnail((300, 300))
            self.image = ImageTk.PhotoImage(decompressed_img)
            self.image_label.config(image=self.image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display decompressed image: {str(e)}")

    def save_preview(self):
        """Save the previewed decompressed image from a .npz file."""
        if self.preview_decoded_img is None:
            messagebox.showerror("Error", "No preview image available to save.")
            return

        # Select output path
        output_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ]
        )
        if not output_path:
            messagebox.showerror("Error", "Please select an output path.")
            return

        # Save the preview image
        quality = self.quality.get()
        output_ext = os.path.splitext(output_path)[1].lower()
        try:
            if output_ext in ('.jpg', '.jpeg'):
                if self.preview_is_grayscale:
                    cv2.imwrite(output_path, self.preview_decoded_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    cv2.imwrite(output_path, cv2.cvtColor(self.preview_decoded_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            elif output_ext == '.png':
                if self.preview_is_grayscale:
                    cv2.imwrite(output_path, self.preview_decoded_img)
                else:
                    cv2.imwrite(output_path, cv2.cvtColor(self.preview_decoded_img, cv2.COLOR_RGB2BGR))
            else:
                messagebox.showerror("Error", "Unsupported output format. Use .jpg, .jpeg, or .png.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preview image: {str(e)}")
            return

        # Update file size info
        compressed_data_size = os.path.getsize(self.input_path.get()) / 1024  # Size in KB
        output_size = os.path.getsize(output_path) / 1024  # Size in KB
        self.file_size_info.set(f"Compressed data size: {compressed_data_size:.2f} KB | Decompressed file size: {output_size:.2f} KB")

        # Show success message
        messagebox.showinfo("Success", "Preview image saved successfully!")

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGCompressorGUI(root)
    root.mainloop()