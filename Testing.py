import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import sys

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {iteration}/{total} images done ({percent}%) {suffix}')
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# Directory containing the images
image_dir = './data/images/'

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Sort image files
image_files.sort()

def process_image(image_file, detect_script, weights, size, model, plate, dont_show, info_flag, database_flag):
    image_path = os.path.join(image_dir, image_file)

    flags = {
        '--plate': plate,
        '--dont_show': dont_show,
        '--info': info_flag,
        '--database': database_flag
    }
    flags_str = ' '.join(flag for flag, enabled in flags.items() if enabled)

    cmd = f'python {detect_script} --weights {weights} --size {size} --model {model} --images {image_path} {flags_str}'

    start_time = time.time()

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    end_time = time.time()

    latency = end_time - start_time

    ocr_result = result.stdout.split('License Plate # : ')[-1].strip().split('\n')[0] if 'License Plate # : ' in result.stdout else None

    confidence = None
    mAP = None

    if info_flag and plate:
        for line in result.stdout.split('\n'):
            if 'Confidence:' in line:
                confidence = float(line.split('Confidence: ')[1].split(',')[0])
            if 'mAP:' in line:
                mAP = float(line.split('mAP: ')[1])

    return image_file, ocr_result, confidence, mAP, latency

detect_script = 'detect.py'
weights = './checkpoints/yolov3-custom-416'
size = '416'
model = 'yolov3'
plate = True
dont_show = True
info_flag = True
database_flag = False

results = []

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

true_positive_yolo = 0
false_positive_yolo = 0
true_negative_yolo = 0
false_negative_yolo = 0

total_images = len(image_files)
total_matches = 0
total_license_plate_recognition = 0

accuracy_yolo = 0
precision_yolo = 0
recall_yolo = 0
f1_score_yolo = 0

print_progress(0, total_images, prefix='Progress:', suffix='Complete', length=50)
update_time = time.time()

for i, image_file in enumerate(image_files):
    image_result, ocr_result, confidence, mAP, latency = process_image(image_file, detect_script, weights, size, model, plate, dont_show,
                                                      info_flag, database_flag)

    actual_plate_number = image_file.split('-')[1].split('.')[0] if image_file.startswith('car-') else None

    plate_number = 'car-' in image_file
    ocr_notEmpty = bool(ocr_result)

    match = actual_plate_number == ocr_result if ocr_result else False

    if plate_number is True:
        if ocr_notEmpty:
            true_positive_yolo += 1
        else:
            false_positive_yolo += 1
            print(f"False Positive YOLO - File: {image_file}, OCR Result: {ocr_result}")
    else:
        if ocr_notEmpty:
            false_negative_yolo += 1
            print(f"False Negative YOLO - File: {image_file}, OCR Result: {ocr_result}")
        else:
            true_negative_yolo += 1

    if match:
        total_matches += 1

    if actual_plate_number:
        total_license_plate_recognition += 1

    results.append({
        'Image': image_file,
        'OCR Result': ocr_result,
        'Confidence': confidence,
        'mAP': mAP,
        'Latency (seconds)': latency,
        'Match': match
    })

    if actual_plate_number is not None:
        if match:
            true_positive += 1
        else:
            false_negative += 1
            print(f"False Negative OCR - File: {image_file}, OCR Result: {ocr_result}")
    else:
        if ocr_result is not None:
            false_positive += 1
            print(f"False Positive OCR - File: {image_file}, OCR Result: {ocr_result}")
        else:
            true_negative += 1

    if time.time() - update_time > 30 or i == total_images - 1:
        current_results = pd.DataFrame(results, columns=['Image', 'OCR Result', 'Confidence', 'mAP', 'Latency (seconds)', 'Match'])

        accuracy_yolo = ((true_positive_yolo + true_negative_yolo) /
                         (true_positive_yolo + true_negative_yolo + false_positive_yolo + false_negative_yolo)) * 100

        precision_yolo = true_positive_yolo / (true_positive_yolo + false_positive_yolo) * 100 \
            if (true_positive_yolo + false_positive_yolo) > 0 else 0

        recall_yolo = true_positive_yolo / (true_positive_yolo + false_negative_yolo) * 100 \
            if (true_positive_yolo + false_negative_yolo) > 0 else 0

        f1_score_yolo = 2 * (precision_yolo * recall_yolo) / (precision_yolo + recall_yolo) \
            if (precision_yolo + recall_yolo) > 0 else 0

        average_latency = current_results['Latency (seconds)'].mean()

        average_ocr_accuracy = (total_matches / total_license_plate_recognition) * 100 if total_license_plate_recognition > 0 else 0

        print_progress(i + 1, total_images, prefix='Progress:', suffix='Complete', length=50)
        print(f"Current Average Latency: {average_latency:.2f} seconds")
        print(f"Current OCR Accuracy (confusion): {average_ocr_accuracy:.2f}%")

        ocrAvg = average_ocr_accuracy
        update_time = time.time()

# Create DataFrame
df_results = pd.DataFrame(results, columns=['Image', 'OCR Result', 'Confidence', 'mAP', 'Latency (seconds)', 'Match'])

# Save results to Excel file
timestamp = time.strftime("%Y%m%d-%H%M%S")
excel_name = f'results/results_{timestamp}.xlsx'

with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Detection Results', index=False)

    avg_df = pd.DataFrame({
        'Metric': ['Average Latency (seconds)', 'OCR Accuracy (%)'],
        'Value': [average_latency, average_ocr_accuracy]
    })
    avg_df.to_excel(writer, sheet_name='Average Metrics', index=False)

    confidence_df = pd.DataFrame({
        'Image': df_results['Image'],
        'Confidence': df_results['Confidence']
    })
    confidence_df.to_excel(writer, sheet_name='Confidence', index=False)

    map_df = pd.DataFrame({
        'Image': df_results['Image'],
        'mAP': df_results['mAP']
    })
    map_df.to_excel(writer, sheet_name='mAP', index=False)

# Visualization
y_true = [actual_plate_number is not None for actual_plate_number in [image_file.split('-')[1].split('.')[0] if '-' in image_file else None for image_file in image_files]]
y_pred = [ocr_result is not None for ocr_result in df_results['OCR Result'].tolist()]

cm = confusion_matrix(y_true, y_pred, labels=[True, False])

confusion_matrix_df = pd.DataFrame(
    cm,
    index=['Actual Positive (True)', 'Actual Negative (False)'],
    columns=['Predicted Positive', 'Predicted Negative']
)

metrics_df_yolo = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy_yolo, precision_yolo, recall_yolo, f1_score_yolo]
})

with pd.ExcelWriter(excel_name, mode='a', engine='openpyxl') as writer:
    confusion_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
    metrics_df_yolo.to_excel(writer, sheet_name='Metrics', index=False)

plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(f'results/confusion_matrix_{timestamp}.png')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='Value', y='Metric', data=metrics_df_yolo, palette='Blues_d')
plt.title('Performance Metrics', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Metric', fontsize=14)
plt.savefig(f'results/performance_metrics_{timestamp}.png')
plt.show()

print(f"Accuracy value for OCR Accuracy chart: {ocrAvg}")

plt.figure(figsize=(8, 6))
plt.bar(['OCR Accuracy'], [ocrAvg], color='blue')
plt.text(0, ocrAvg + 1, f'{ocrAvg:.2f}%', ha='center', fontsize=12)
plt.title('OCR Accuracy', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(0, 100)
plt.savefig(f'results/ocr_accuracy_graph_{timestamp}.png')
plt.show()

print(f"OCR Accuracy: {ocrAvg:.2f}%")
print(f"Average Latency: {average_latency:.2f} seconds")
print(f"Results saved in {excel_name}")
