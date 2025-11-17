# 👗 Fashion Dupe Detection System

A deep learning-based fashion item classifier and duplicate detector using PyTorch and ResNet50.

## 🎯 Features

- **Single Image Classification**: Classify fashion items into categories (men/women/footwear)
- **Dupe Detection**: Compare two images and detect if they are duplicate/similar products
- **Interactive UI**: Built with Streamlit for easy deployment
- **High Accuracy**: Uses transfer learning with ResNet50 pretrained on ImageNet

## 🏗️ Model Architecture

- **Backbone**: ResNet50 (pretrained)
- **Custom Head**: Fully connected layers with BatchNorm and Dropout
- **Embedding Layer**: 512-dimensional feature vectors for similarity matching
- **Training**: PyTorch with Adam optimizer and learning rate scheduling

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit
- See `requirements.txt` for full list

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fashion-dupe-detection.git
cd fashion-dupe-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the trained model file:
   - Place `best_fashion_dupe_model.pth` in the root directory

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

**Single Image Classification Mode:**
1. Select "Single Image Classification" in the sidebar
2. Upload a fashion item image
3. View the predicted category and confidence scores

**Dupe Detection Mode:**
1. Select "Dupe Detection (Compare 2 Images)" in the sidebar
2. Upload two fashion item images
3. Adjust the similarity threshold if needed
4. Click "Compare Images"
5. View similarity score and dupe verdict

## 📁 Project Structure

```
fashion-dupe-detection/
├── app.py                              # Streamlit application
├── model.py                            # Model architecture definition
├── utils.py                            # Utility functions
├── requirements.txt                    # Python dependencies
├── best_fashion_dupe_model.pth        # Trained model weights (you need this!)
├── Fashion_Dupe_Detection_Complete-2.ipynb  # Training notebook
└── README.md                           # This file
```

## 🎓 Model Training

The model was trained using the notebook `Fashion_Dupe_Detection_Complete-2.ipynb` with:
- **Dataset**: Fashion items organized into categories
- **Training Strategy**: Transfer learning with ResNet50
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Best Validation Accuracy**: Achieved during training

## 🔧 Customization

### Update Class Names

In `app.py`, modify the `class_names` list to match your dataset:
```python
class_names = ['footwear', 'men', 'women']  # Update as needed
```

### Adjust Similarity Threshold

Change the default threshold in the sidebar or modify it in `utils.py`:
```python
def is_dupe(similarity_score, threshold=75):  # Adjust threshold here
```

## 🐛 Troubleshooting

### Model Loading Error
**Error**: `FileNotFoundError: best_fashion_dupe_model.pth`

**Solution**: Make sure your trained model file is in the same directory as `app.py`. If you haven't trained the model yet, run the training notebook first.

### CUDA/GPU Issues
If you encounter GPU-related errors, the app will automatically fall back to CPU. To force CPU usage:
```python
device = torch.device('cpu')
```

### Image Upload Issues
Make sure uploaded images are in RGB format (PNG, JPG, JPEG). The app automatically converts images if needed.

## 📊 Performance

- **Classification Accuracy**: ~XX% (update with your results)
- **Inference Time**: < 1 second per image (CPU)
- **Model Size**: ~100 MB

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Your Name - Week 3 Project

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit for the amazing web app framework
- ResNet50 architecture by He et al.

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

Made with ❤️ and PyTorch
