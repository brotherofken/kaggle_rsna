# kaggle_rsna

## How to

Solution is under heavy development and baseline not working yet. Use at your own risk and expect waste of time.

### 1. Preprocess data

Test data
```
python convert_data_tf_format.py --input_images_path rsna_data/stage_1_test_images --input_labeling_path rsna_data/stage_1_test_images.csv --output_path rsna_data_preprocessed/stage_1_test_images.1.tfrecord --threads 7
```
Train data
```
python convert_data_tf_format.py --input_images_path rsna_data/stage_1_train_images --input_labeling_path rsna_data/stage_1_train_labels.csv --output_path rsna_data_preprocessed/stage_1_train_images.tfrecord --threads 7
```

### 2. Train baseline
Run as follows:
```
python ${TENSORFLOW_MODELS_REPO}/research/object_detection/model_main.py --alsologtostderr --pipeline_config_path models/baseline_model/ssd_mobilenet_v1_focal_loss.config --model_dir models/baseline_model
```
where
- ${TENSORFLOW_MODELS_REPO} - path to your copy of https://github.com/tensorflow/models