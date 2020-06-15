
# Text line recognition using a BLSTM neural network

In this work we propose a neural network model to transcribe pages from the original Gutenberg's Bible using CNNs and BLSTMs. After expanding the dataset to add the Exodus book by manually aligning the text transcription to the pages of the book, we used this data to refine our model. Then we conducted several experiments to test other models found in literature in order to compare the results.

### Prerequisites

All the requirements needed for the project are available in the `requirements.txt` file. You can install them using

```
pip install -r requirements.txt
```

### Dataset

The dataset could be found at the following [link](https://drive.google.com/file/d/1dHIG8LPvInPb4hNakitM08kApL6WIOIM/view?usp=sharing). It contains the images of the bible and the transcription of the first two books: Genesis and Exodus.

## Paper

You can find a detailed [paper](../master/Paper/D_DM_Project.pdf) in English about the entire project, the experiments and the results here. Following you can find some snippet of the most important parts

## Detailed Model View

We used a neural network model composed by

 - 6 Convolutional Neural Network layers
 - 2 Bidirectional LSTM layers
 - 1 CTC layer
 
![Detailed model view](../master/Paper/htr_model_detailed.png)

## Results

To better understand the experiments that lead to these results, consider the [paper](../master/Paper/D_DM_Project.pdf). Below are presented the learning curves of the five conducted experiments for both training and validation. The lighter-colored curve represents the training while the darker one the validation

![Learning curves results](../master/Paper/losses-2.png)

## Authors

* **Federico Palai** - [GitHub](https://github.com/palai103)

* **Federico Tammaro** - [GitHub](https://github.com/sfullez)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
