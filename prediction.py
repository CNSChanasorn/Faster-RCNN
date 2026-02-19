import matplotlib.pyplot as plt
import matplotlib.patches as patches

def predict_and_draw(model, image, class_names, threshold=0.5):
    # Run YOLO inference
    results = model(image, conf=threshold, verbose=False)
    
    class_counts = {name: 0 for name in class_names[1:]}

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    ax.axis('off')

    # Extract detections from YOLO results
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                if scores[i] > threshold:
                    box = boxes[i]
                    label_idx = labels[i]
                    
                    # Ensure label is within class_names range
                    if label_idx < len(class_names):
                        class_name = class_names[label_idx]
                        
                        if class_name in class_counts:
                            class_counts[class_name] += 1

                        rect = patches.Rectangle(
                            (box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
                        ax.text(box[0], box[1], f"{class_name} {scores[i]:.2f}", 
                                 color='white', verticalalignment='top', 
                                 bbox={'color': 'red', 'alpha': 0.5, 'pad': 0})

    return fig, class_counts