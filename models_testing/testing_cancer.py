from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load model
model = YOLO('my_model3.pt')

# Predict and get plotted image
results = model.predict('C:\\Users\\kumar\\Downloads\\AI-BASED-HEART-FAILURE-PREDICTION\\heart_failure\\mouth_cancer_images\\Screenshot 2025-04-24 033706.png')

# Extract the plotted image (with bounding boxes)
plotted_img = results[0].plot()  # Returns a numpy array (BGR format)

# Convert BGR to RGB for matplotlib
plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(10, 8))
plt.imshow(plotted_img_rgb)
plt.axis('off')  # Hide axes
plt.show()