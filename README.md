# Introduction

**Several neural models for Chinese word segment**.

This work was completed when I was a natural language processing engineer in *Baidu*.
  
Please kindly give a star if you use this code. 

## Requirements

* Python >= 3.2
* Numpy >= 1.11
* Panda >= 0.17
* Keras 1.0

## Usage

* Preprocess data
```bash
python preprocess.py
```
* Generate train and test dataset
```bash
python dataset_test.py
```

* Generate train and test dataset with location
```bash
python with_location_dataset.py
```

## Training
* Train the models

* Train LSTM: ```python word_seg_lstm.py```

* Train CNN: ```python word_seg_cnn.py```

* Train shared LSTM: ```python share_lstm.py```

* Train location-based LSTM: ```python with_locations_lstm.py```

