# LSTM-IMDB
Long and short term memory neural networks to classify text entries according to whether the opinion is positive or negative.
Trained with the IMDB Dataset.

## Description

IMDB dataset has 50K movie reviews for natural language processing or Text analytics.

The model consists of :
* an Embedding layer to encode the data.  
* an LSTM layer composed of 128 neurons with a Dropout of 0.2 to avoid overlearning.  
* A layer composed of one neuron with a sigmoid activation function to give the opinion on the tested sentence.  

* The resulting model is saved in the imdb.h5 file.  
* The accuracy of the model is 84 %.  

## Getting Started

### Dependencies

* IMDB Dataset
* Python
* Keras with Tensorflow backend

### Executing program

* Run imdb.py
```
python imdb.py
```

## Author


KARUNAKARAN Nithushan

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

