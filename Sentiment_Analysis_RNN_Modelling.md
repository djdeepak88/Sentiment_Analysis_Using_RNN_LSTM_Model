In this project, I have built a deep learning model to classify the sentiment of messages from StockTwits, a social network for investors and traders.
This model will be able to predict if any particular message is positive or negative. From this, we can generate a signal of the public sentiment for various ticker symbols.

The Dataset comprises of the twits from the StockTwits and each being given a sentiment from -2 to +2.
The task was to train a RNN model on this data to predict the sentiment of a given twit ranging from Negative to Neutral
to Positive Sentiment.

We used a RNN-LSTM Model (2048->1024->5(Output Probability Class)) with LogSoftMax activation function
since we have 5 categories of sentiments to predict on.
The Preprocessing involves removing punctuations, stopwords, numeric words and then tokenize the text
and assign the numerical values to each unique word.
The Sentiments were rescaled from -2 to +2 to 0 to 4. 

LSTMs are the type of the recurrent neural nets. In recurrent neural nets weight parameters are shared between the hidden units. That means information given to the hidden unit at time t is not only coming from input unit but also from the hidden units of previous time stamps. So the input nodes and previous hidden states are concatenated together and then multiplied with weights. As a result, back-propagation through time is carried out as the inputs from the previous timestamps are also considered. The sequential learning by sharing the weights is carried within the sequence length hyperparameter in the RNN layer.