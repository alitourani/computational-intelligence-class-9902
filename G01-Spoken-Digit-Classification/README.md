# Spoken Digit Classification

## Team Members:
- Fatemeh Kamani (970122680033)
- Arash Alipour (9712268100)
- Amir Abbas Nasiri (980122680047)
------------

## Description:


Explaining video link:</br>
https://drive.google.com/file/d/1kc8JfOezi_YX4T0YuMrl0hpDe6as9T5D/view?usp=sharing</br>
</br>
In this project we used the data set collected from:</br>
https://github.com/Jakobovski/free-spoken-digit-dataset/tree/master/recordings</br>
</br>
this data set has 3000 audio files with 6 reading 10 digits meaning each person read each digit 50 times</br>
</br>
then we used functions in "remove_silences.ipynb" file to remove the silence from these audios</br>
</br>
you can access our audio files in our drive in "Audio/" folder:</br>
https://drive.google.com/drive/folders/1uMnuZp2s2cafBznaZ7HjbD-U72SlQ27G?usp=sharing</br>
</br>
After that we used "Spoken_Digits_Classification.ipynb" to extract features form the voices in “cut_silence/” using librosa mfccs function with 30 samples in each audio file</br>
so now we have data features and labels of our data_set</br>
</br>
then we chopped the 2808 audio files into 3 portions of training, validation and test set (80%,10%,10%)</br>
</br>
then we used a MLP Network with 4 layer: </br>
input: 30n</br>
hidden1: 120n, af=’relu’</br>
hidden2: 120n, af=’relu’</br>
out: 10 n</br>
we used 25 epochs</br>
</br>
(these values were calculated through looking at test evaluation results and plots of “loss” and “accuracy” epoch-history generated after the end of trainings)</br>
</br>
finally we achieved 90% accuracy with the above values</br>
