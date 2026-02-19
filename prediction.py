import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def predict_and_draw(model, image, class_names, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        prediction = model(img_tensor.to(device))

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    class_counts = {name: 0 for name in class_names[1:]}

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    ax.axis('off')

    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]
            label_idx = labels[i]
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