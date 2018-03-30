import numpy as np
import nltk
import itertools
import RNNNumpy as rnp
import io
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
# http://www.nltk.org/book/ch01.html
# https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
VOCAB_SIZE = 8000
MIN_FREQUENCY = 2;
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
START_TOKEN = "START_TOKEN"
END_TOKEN = "END_TOKEN"

raw_data = open('ryan.txt','r').read()

sentances = nltk.sent_tokenize(raw_data.lower().decode('utf-8'))
sentances = [START_TOKEN + ' ' + x + ' ' + END_TOKEN for x in sentances]
tok_sentances = [nltk.word_tokenize(s) for s in sentances]

word_freq = nltk.FreqDist(itertools.chain(*tok_sentances))
vocabulary = [x[0] for x in word_freq.items() if x[1] >=MIN_FREQUENCY ] + [UNKNOWN_TOKEN]
vocabulary = vocabulary[:min(len(vocabulary),VOCAB_SIZE)]

word_to_index = dict( [(w,i) for i,w in enumerate(vocabulary)])


for i,sent in enumerate(tok_sentances):
    tok_sentances[i] = [w if w in vocabulary else UNKNOWN_TOKEN for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tok_sentances])
Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tok_sentances])
# X_train = [word for sent in tok_sentances for word in sent]



np.random.seed(11)
model = rnp.RNNNumpy(len(vocabulary))
# o, s = model.forward_propagation(X_train[10])
# print X_train[10]
# print " ".join([vocabulary[x] for x in X_train[25]])
# print o.shape
# print o
# predictions = model.predict(X_train[25])
# print predictions
# print predictions.shape

# print "Expected Loss for random predictions: %f" % np.log(len(vocabulary))
# print "Actual loss: %f" % model.calculate_loss(X_train[:1000], Y_train[:1000])

# model2 = rnp.RNNNumpy(100,10,1000)
# model.gradient_check([0,1,2,3],[1,2,3,4])

# start = int(round(time.time() * 1000))
# model.sgd_step(X_train[10], Y_train[10], 0.005)
# end = int(round(time.time() * 1000))
# print end - start
#,X_train, Y_train, learning_rate=.005, nepoch=100, evaluate_loss_after=5):

model.train_with_sgd(X_train,Y_train,.005,30,1)

def gen_sent(model):
    new_sentance = [word_to_index[START_TOKEN]]
    while not new_sentance[-1] == word_to_index[END_TOKEN] and len(new_sentance) < 8:
        next_word_probs, _ = model.forward_propagation(new_sentance)
        sample_word = word_to_index[UNKNOWN_TOKEN]
        while sample_word == word_to_index[UNKNOWN_TOKEN]:
            samples = np.random.multinomial(1,next_word_probs[-1])
            sample_word = np.argmax(samples)
        new_sentance.append(sample_word)
    sentance_str = [vocabulary[x] for x in new_sentance]
    return sentance_str

sents = [];
for i in range(10):
    sents.append(gen_sent(model))
for sent_arr in sents:
    print " ".join([x.encode("utf-8")for x in sent_arr[1:]])