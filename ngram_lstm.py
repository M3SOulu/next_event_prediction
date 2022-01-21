import numpy as np
import os
from nltk import ngrams
from pandas.core.frame import DataFrame
import os
import time
import random
import pickle
import math
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from tensorflow import keras
from collections import Counter
from collections import defaultdict  

#Setting pointing to where one wants to load and save data. 
os.chdir("/home/ubuntu/next_event_prediction/data")

#Global variables
_ngrams_ = 5
_start_ ="SoS" #Start of Sequence used in padding the sequence
_end_ = "EoS" #End of Sequence used in padding the sequence
#More clo
n_gram_counter = Counter()
n_gram_counter_1 = Counter() 
n1_gram_dict = defaultdict() # to keep mappings of possible following events e1 e2 -> e1 e2 e3, e1 e2 e4, 
n1_gram_winner = dict() #What is the event n following n-1 gram, i.e. the prediction ?


def create_ngram_model(train_data):
    global n_gram_counter, n_gram_counter_1
    ngrams = list()
    ngrams_1 = list()
    for seq in train_data:
        seqs, seqs_1 = slice_to_ngrams(seq)
        ngrams.extend(seqs)
        ngrams_1.extend(seqs_1)
    n_gram_counter += Counter (ngrams)
    n_gram_counter_1 += Counter (ngrams_1)

    for idx, s in enumerate(ngrams):
        #dictionary for faster access from n-1 grams to n-grams, e.g. from  [e1 e2 e3] -> [e1 e2 e3 e4]; [e1 e2 e3] -> [e1 e2 e3 e5] etc...
        n1_gram_dict.setdefault(ngrams_1[idx],[]).append(s)
        #precompute the most likely sequence following n-1gram. Needed to keep prediction times fast
        if (ngrams_1[idx] in n1_gram_winner): #is there existing winner 
            n_gram = n1_gram_winner[ngrams_1[idx]]
            if (n_gram_counter[n_gram] < n_gram_counter[s]): #there is but we are bigger replace
                n1_gram_winner[ngrams_1[idx]] = s
        else: 
            n1_gram_winner[ngrams_1[idx]] = s #no n-1-gram key or winner add a new one...

#Produce required n-grams. E.g. With sequence [e1 ... e5] and _ngrams_=3 we produce [e1 e2 e3], [e2 e3 e4], and [e3 e4 5] 
def slice_to_ngrams (seq):
    #Add SoS and EoS
    #with n-gram 3 it is SoS SoS E1 E2 E3 EoS
    #No need to pad more than one EoS as the final event to be predicted is EoS
    seq = [_start_]*(_ngrams_-1) +seq+[_end_]
    ngrams = list()
    ngrams_1 = list()
    for i in range(_ngrams_, len(seq)+1):#len +1 because [0:i] leaves out the last element 
        ngram_s = seq[i-_ngrams_:i]
        # convert into a line
        line = ' '.join(ngram_s)
        # store
        ngrams.append(line)
        ngram_s_1= seq[i-_ngrams_:i-1]
        line2 = ' '.join(ngram_s_1)
        # store
        ngrams_1.append(line2)
    return ngrams, ngrams_1


# Return two anomaly scores as in the paper
# Ano score per line (i.e. given the previous lines how probable is this line). 
# And n of occurences per line seen in the past
def give_ano_score (seq):
    seq_shingle, seq_shingle_1 = slice_to_ngrams(seq)
    scores = list()
    for s in seq_shingle:
        scores.append(n_gram_counter [s])
    scores_1 = list()
    for s in seq_shingle_1:
        scores_1.append(n_gram_counter_1 [s])

    #Remove 0s from n1 gram list to get rid of division by zero. 
    # If n-1 gram is zero following n-gram must be zero as well so it does not effect the results
    scores_1 = [1 if i ==0 else i for i in scores_1]
    #Convert n-gram freq counts to probs of n-gram given n-gram-minus-1
    scores_prop = np.divide(np.array(scores), np.array(scores_1))
    scores_abs = np.array(scores)
    return (scores_prop, scores_abs)


def load_pro_data():
    pro_x = np.load("profilence_x_data.npy", allow_pickle=True)
    pro_y = np.load("profilence_y_data.npy", allow_pickle=True)

    pro_y = pro_y == 1
    abnormal_test = pro_x[pro_y]

    pro_x_normal = pro_x[~pro_y]
    from nltk import ngrams

    lengths = list()
    for seq in pro_x_normal:
        lengths.append(len(seq))
    #zeros = np.array([True if i ==0 else False for i in lengths])
    #pro_x_normal = pro_x_normal[~zeros]
    #Remove the short logs less than 10000
    ten_k_lenght = np.array([True if i >= 10000 else False for i in lengths])
    pro_x_normal = pro_x_normal[ten_k_lenght]
    normal_data = pro_x_normal
    return normal_data, abnormal_test



def load_hdfs_data():
    hdfs_x = np.load("hdfs_x_data.npy", allow_pickle=True)
    hdfs_y = np.load("hdfs_y_data.npy", allow_pickle=True)

    hdfs_y = hdfs_y == 1

    hdfs_x_normal = hdfs_x[~hdfs_y]
    abnormal_test = hdfs_x[hdfs_y]
    normal_data = hdfs_x_normal
    return normal_data, abnormal_test


#Reset global n-gram variables. Used when creating multiple n-gram models
def reset_globals():
    global n_gram_counter, n_gram_counter_1, n1_gram_dict, n1_gram_winner
    n_gram_counter = Counter()
    n_gram_counter_1 = Counter()
    from collections import defaultdict
    n1_gram_dict = defaultdict() # to keep mappings of possible following events e1 e2 -> e1 e2 e3, e1 e2 e4, 
    n1_gram_winner = dict()
    #sequences = list()
    #sequences_1 = list()



def create_LSTM_model(ngrams, vocab_size, share_of_data=1):
    #If we want to use less than 100% of data select samples. I am not sure this is ever used
    if (share_of_data < 1):
        select = int(len(ngrams) * share_of_data)
        ngrams = random.sample(ngrams, select)

    # How many dimensions will be used to represent each event. 
    # With words one would use higher values here, e.g. 200-400
    # Higher values did not improve accuracy but did reduce perfomance. Even 50 might be too much
    dimensions_to_represent_event = 50
    
    model = Sequential()
    model.add(Embedding(vocab_size, dimensions_to_represent_event, input_length=_ngrams_-1))
    # We will use a two LSTM hidden layers with 100 memory cells each. 
    # More memory cells and a deeper network may achieve better results.
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Loop needed as Office PC would crash in the to_categorixal with Profilence data set as it out of memory. 
    #TODO: Do we need a loop when using CSC HW?
    loop_variable = 50000
    for x in range(0, len(ngrams), loop_variable):
        print(f'loop with x= {x}. / {len(ngrams)}')
        ngrams0 = np.array(ngrams[x:x+loop_variable])
        X, y = ngrams0[:,:-1], ngrams0[:,-1]
        y = to_categorical(y, num_classes=vocab_size)
        #Modify batch_size and epoch to influence the training time and resulting accuracy. 
        history = model.fit(X, y, validation_split=0.05, batch_size=1024, epochs=10, shuffle=True).history
    
    return model

# We need to change events e1 e2 e3 to numbers for the DL model so they are mapped here, e.g. e1 -> 137, e2 -> 342 
def sequences_to_dl_ngrams (train_data):
    ngrams = list() #ngrams= []
    for seq in train_data:
        t_ngrams, t_ngrams_1 = slice_to_ngrams(seq)
        ngrams.extend(t_ngrams)
    tokenizer = Tokenizer(oov_token=1)    
    tokenizer.fit_on_texts(ngrams)
    ngrams_num = tokenizer.texts_to_sequences(ngrams)
    vocab_size = len(tokenizer.word_index) + 1
    return ngrams, ngrams_num, vocab_size, tokenizer

#Gives N-gram predictions
def give_preds (seq):
    seq_shingle, seq_shingle_1 = slice_to_ngrams(seq)
    #   print(seq_shingle)
    correct_preds = list()
    for s in seq_shingle:
        to_be_matched_s =  s.rpartition(' ')[0]
        #print("to be matched " + to_be_matched_s)
        if (to_be_matched_s in n1_gram_dict):
            winner = n1_gram_winner[to_be_matched_s]
            if (winner == s):
                correct_preds.append(1)
                #print("correct")
            else: 
                correct_preds.append(0)
                #print("incorrec predic")
        else:
            correct_preds.append(0)
            #print("no key")
    return correct_preds

#LSTM prediction per sequence. Typically called from loop that with HDFS is not efficient
def give_preds_lstm (seq):
    seq_shingle, seq_shingle_1 = slice_to_ngrams(seq)
    seq_shingle_num = lstm_tokenizer.texts_to_sequences(seq_shingle)
    seq_shingle_num_np = np.array(seq_shingle_num)
    seq_shingle_num_1 = seq_shingle_num_np[:,:-1]
    seq_shingle_truth = seq_shingle_num_np[:,-1]

    #predicted_sec = model.predict(seq_shingle_num_1)
    predicted_sec = model.predict(seq_shingle_num_1,verbose=1, batch_size=4096)
    predicted_events = np.argmax(predicted_sec, axis=1)

    correct_preds = seq_shingle_truth == predicted_events
    return correct_preds

#LSTM predictions with multiple sequences packed in numpy array 
def give_preds_lstm_2 (sequences, b_size=4096):
    seq_shingle = list()
    #check if this is an array of sequences
    start_s = time.time()
    if (isinstance(sequences, np.ndarray)):
        for s in sequences:
            temp_seq_shingle, temp_seq_shingle_1 = slice_to_ngrams(s)
            seq_shingle.extend(temp_seq_shingle)
    else: #if not numpy array then as 
        seq_shingle, seq_shingle_1 = slice_to_ngrams(sequences)
    end_s = time.time()
    print("Shingle creation took", end_s - start_s)
    start_s = time.time()
    seq_shingle_num = lstm_tokenizer.texts_to_sequences(seq_shingle) #do this before slice to n-grams
    end_s = time.time()
    print("lstm_tokenizer took", end_s - start_s)
    seq_shingle_num_np = np.array(seq_shingle_num)
    seq_shingle_num_1 = seq_shingle_num_np[:,:-1]
    seq_shingle_truth = seq_shingle_num_np[:,-1]

    #predicted_sec = model.predict(seq_shingle_num_1)
    start_s = time.time()
    predicted_sec = model.predict(seq_shingle_num_1,verbose=1, batch_size=b_size)
    end_s = time.time()
    print("prediction took", end_s - start_s)
    #predicted_sec = model.predict(seq_shingle_num_1, verbose=1, use_multiprocessing = True, max_queue_size=100,workers=4)
    predicted_events = np.argmax(predicted_sec, axis=1)

    correct_preds = seq_shingle_truth == predicted_events
    return correct_preds




# END of Functions-------------------------------------------------------------------------------------------------------------------
# What follows should executed line-by-line 
#RQ0 Demo case of metrics in paper shown in the final table-------------------------------------------------------------------------------------------
normal_data, abnormal_test = load_hdfs_data()
_ngrams_=5
create_ngram_model(normal_data)

ab_failure = list( abnormal_test[2] ) #1st fail  is FFWH  2nd is WF 3rd is the first long
ano_score = give_ano_score (ab_failure)

for i in range(len(ab_failure)):
    print(ab_failure[i]," ", ano_score[1][i], " ", ano_score[0][i])
    if (i+1 == len(ab_failure)):
        print("EoS ", ano_score[1][i], " ", ano_score[0][i])

print (ano_score[1])
np.average(ano_score[0])
np.percentile(ano_score[1],5)
len(ano_score[0])


#RQ0 Some basic stats for the paper e.g. number of n-grams in data---------------------------------------------------
normal_data, abnormal_test = load_pro_data()
normal_data, abnormal_test = load_hdfs_data()
_ngrams_=1
ngrams = list()
for seq in normal_data:
    seqs, seqs_1 = slice_to_ngrams(seq)
    ngrams.extend(seqs)
    
ngrams = np.array(ngrams)
win_unique, win_counts = np.unique(ngrams, return_counts=True)
win_counts[np.argmax(win_counts)]

for i in range(10):
    _ngrams_ = i+1
    start_s = time.time()
    ngrams = list()
    for seq in normal_data:
        seqs, seqs_1 = slice_to_ngrams(seq)
        ngrams.extend(seqs)
    win_unique, win_counts = np.unique(ngrams, return_counts=True)
    end_s = time.time()
    print ("N-grams: ",_ngrams_," Unique:", len(win_unique), "Done in:", end_s-start_s)    



# RQ1---------------------------------------------------------------------------------------------------
# Data loading

#Select variable on which data set to load
data="hdfs"
data="pro"

if (data=="hdfs"):
    print("Setting data to HDFS")
    normal_train = np.loadtxt('split_normal_hdfs_train.txt') #load split
    normal_data, abnormal_test = load_hdfs_data() #load data
elif(data=="pro"):
    print("Setting data to PRO")
    normal_train = np.loadtxt('split_normal_pro_train.txt') #load split
    normal_data, abnormal_test = load_pro_data() #load data"
normal_train = np.array(normal_train, dtype=bool)
normal_test = normal_data[~normal_train]

#Creating split. Uncomment if new split needed. Currently we just load the pre-saved split
#train_i = np.random.choice(normal_data.shape[0], np.floor_divide(normal_data.shape[0],2), replace=False)
#normal_train = np.isin(range(normal_data.shape[0]), train_i)
#save data
#np.savetxt('split_normal_pro_train.txt', normal_train, fmt='%d') #PRO
#np.savetxt('split_normal_hdfs_train.txt', normal_train, fmt='%d') #HDFS


#---Create models
#ngram---------------------------------------------------------
_ngrams_ = 5
#ngram model
start_s = time.time()
create_ngram_model(normal_data[normal_train])
end_s = time.time()
print("ngram with ngrams:", _ngrams_, "done in", end_s - start_s)

#lstm model-load/creation---------------------------------------------
create_model = "yes"
create_model = "no"
if (create_model=="yes"):
    start_s = time.time()
    lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, lstm_tokenizer = sequences_to_dl_ngrams(normal_data[normal_train])
    model = create_LSTM_model(lstm_ngrams_num, lstm_vocab_size, share_of_data=1)
    end_s = time.time()
    print("lstm with ngrams:", _ngrams_, "done in", end_s - start_s)
    if (data=="hdfs"):
        #load save model
        #model.save("ngram5_lstm_hdfs_50_normal_all_data_20_11_2021")
        #model.save("ngram5_lstm_hdfs_50_normal_all_data_14_01_2022")
        model.save("ngram5_lstm_hdfs_50_normal_all_data_CURRENT_DATE")
        # saving tokenizer
        with open('tokenizer_5_lstm_hdfs_50__CURRENT_DATE.pickle', 'wb') as handle:
            pickle.dump(lstm_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif(data=="pro"):
        #Model save / load
        #model.save("ngram5_lstm_pro_50_normal_all_data_14_01_22")
        model = keras.models.load_model("ngram5_lstm_pro_50_normal_all_data_CURRENT_DATE")
        # saving tokenizer
        with open('tokenizer_5_lstm_pro_50_CURRENT_DATE.pickle', 'wb') as handle:
            pickle.dump(lstm_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
elif(create_model=="no"):
    if (data=="hdfs"):
        model = keras.models.load_model("ngram5_lstm_hdfs_50_normal_all_data_14_01_2022")
        with open('tokenizer_5_lstm_hdfs_50_14_01_22.pickle', 'rb') as handle:
            lstm_tokenizer = pickle.load(handle)
    elif(data=="pro"):
        model = keras.models.load_model("ngram5_lstm_pro_50_normal_all_data_14_01_22")
        with open('tokenizer_5_lstm_pro_50_14_01_22.pickle', 'rb') as handle:
            lstm_tokenizer = pickle.load(handle)
    

# LSTM Prediction------------------------------------------------------------------
#LSTM much faster with HDFS as one predict call instead of loop
start_s = time.time()
lstm_preds_all = list()
if (data=="hdfs"):
    lstm_preds_all = give_preds_lstm_2(normal_test)
elif (data=="pro"):#Cannot do all pro data in one pass runs out of memory at 15gigs. Split to five calls
    for i in range(int(len(normal_test)/10)):
        lstm_preds_t = give_preds_lstm_2(normal_test[i:i+10])
        lstm_preds_all.extend(lstm_preds_t)  
end_s = time.time()
print("prediction time lstm with ngrams:", _ngrams_, "done in", end_s - start_s)
np.mean(lstm_preds_all)


#LSTM with loop. Warning SLOW for HDFS!
start_s = time.time()
print("len is", len(normal_test))
progress = math.floor(len(normal_test)/10)
lstm_preds = list()
for i in range(len(normal_test)):
    if (i % progress ==0):    #as it is slow print line every 10% of progress elements
        print ("loop is at:",i,"/",len(normal_test))
    preds_2 = give_preds_lstm(normal_test[i])
    lstm_preds.append(preds_2)
end_s = time.time()
print("prediction time lstm with ngrams:", _ngrams_, "done in", end_s - start_s)
#---------------------------------------------------
#Studying results of lstm prediction
lstm_preds_means = list()
for preds in lstm_preds:
    lstm_mean = np.mean(preds)
    lstm_preds_means.append(lstm_mean)
    #print (np.mean(lstm_mean))
print("Mean of means", np.mean(lstm_preds_means))


#ngram prediction-------------------------------------------
#ngram test with loop
ngram_preds = list()
start_s = time.time()
for normal_s in normal_test:
    preds = give_preds(normal_s)
    ngram_preds.append(preds)
    #print(".")
end_s = time.time()
print("prediction time ngram with ngrams:", _ngrams_, "done in", end_s - start_s)
#ngram investigate
ngram_preds_means = list()
for preds in ngram_preds:
    ngram_mean = np.mean(preds)
    ngram_preds_means.append(ngram_mean)
    #print (np.mean(lstm_mean))
print("Mean of means", np.mean(ngram_preds_means))
ngram_preds_means

#Joint prediction again in CSC some crashes---------------------------------------------
lstm_preds = list()
ngram_preds = list()
for normal_s in normal_test:
    preds = give_preds(normal_s)
    ngram_preds.append(preds)
    preds_2 = give_preds_lstm(normal_s)
    lstm_preds.append(preds_2)
    print("Ngram accuracy:",np.mean(preds), "LSTM accuracy", np.mean(preds_2))
#save and load predictions
# with open("lstm_hdfs_preds.txt", "wb") as fp:   #Pickling
#     pickle.dump(lstm_preds, fp)
# with open("ngram_hdfs_preds.txt", "wb") as fp:   #Pickling
#     pickle.dump(ngram_preds, fp)
with open("lstm_hdfs_preds.txt", "rb") as fp:   # Unpickling
    lstm_preds = pickle.load(fp)
with open("ngram_hdfs_preds.txt", "rb") as fp:   # Unpickling
    ngram_preds = pickle.load(fp)        

#investigate predictions-both------------------------
#here we can also do sequence by sequence investigation computs wins, ties, losses
lstm_sum= 0
tie_sum = 0
ngram_sum = 0
lstm_preds_means = list()
ngram_preds_means = list()
for i in range(len(lstm_preds)):
    lstm_mean = np.mean(lstm_preds[i]) 
    ngram_mean = np.mean(ngram_preds[i])
    lstm_preds_means.append(lstm_mean)
    ngram_preds_means.append(ngram_mean)
    if (math.isclose(lstm_mean, ngram_mean, rel_tol=1e-4)):
    #if ( lstm_mean == ngram_mean):
        tie_sum = tie_sum +1
    elif (lstm_mean> ngram_mean):
        lstm_sum = lstm_sum +1    
    else:
        ngram_sum = ngram_sum +1    

np.mean(lstm_preds_means)
np.mean(ngram_preds_means)
tie_sum
lstm_sum
ngram_sum



#RQ2---ORIG------------------------------------------------------------------

#RQ2 for analysing both data sets
ngram_preds_all = list()
for i in range (10):
    reset_globals()
    _ngrams_ = i+1
    start_s = time.time()
    for j in range(5):
        reset_globals()
        create_ngram_model(normal_data[normal_train])
    end_s = time.time()
    print("ngrams model create ", _ngrams_, " done in ", end_s - start_s, "for each", (end_s - start_s)/5)
    ngram_preds = list()
    #ngram test
    start_s = time.time()
    for j in range(5):
        for normal_s in normal_test:
            preds = give_preds(normal_s)
            ngram_preds.append(preds)
            #print(".")
    end_s = time.time()
    print("ngrams infer ", _ngrams_, " done in ", end_s - start_s, "for each", (end_s - start_s)/5)
    ngram_preds_all.append(ngram_preds)


import math
ngram_preds_all_mean = list()
for i in range (10):
    ngram_preds_means = list()
    ngram_preds = ngram_preds_all[i]
    for j in range(len(ngram_preds)):
        ngram_mean = np.mean(ngram_preds[j])
        ngram_preds_means.append(ngram_mean)
    #    if (math.isclose(lstm_mean, ngram_mean, rel_tol=1e-4)):
        #if ( lstm_mean == ngram_mean):
    #        tie_sum = tie_sum +1
    #    elif (lstm_mean> ngram_mean):
    #        lstm_sum = lstm_sum +1    
    #    else:
    #       ngram_sum = ngram_sum +1    
    ngram_preds_all_mean.append(ngram_preds_means)



for i in range (10):
    print(np.mean(ngram_preds_all_mean[i]))

ngram_preds_all_mean_df = DataFrame(ngram_preds_all_mean)
winners = list()
ties = 0
for i in range(len(ngram_preds_all_mean[1])):
    #GO over column wise and pick winning n-gram model
    if (i % 1000 ==0):
        print(i)
    if (np.sum(ngram_preds_all_mean_df.iloc[:,i] == np.max(ngram_preds_all_mean_df.iloc[:,i])) > 1):
        ties = ties +1
    else:     
        winners.append(np.argmax(ngram_preds_all_mean_df.iloc[:,i]))

winners = np.array(winners)
win_unique, win_counts = np.unique(winners, return_counts=True)
print(ties)

your_list = np.asarray([7, 6, 5, 7, 6, 7, 6, 6, 6, 4, 5, 6]) #MM: I have no clue what this is.
winners = np.flatnonzero(your_list == np.max(your_list))
winners
array([0, 3, 5])

np.sum(your_list == np.max(your_list))

np.sum(ngram_preds_all_mean_df.iloc[:,2] == np.max(ngram_preds_all_mean_df.iloc[:,2]))

#RQ3-LSTM-Loop-EXTENDED-----------LSTM_vs_ngram----Pro---------------------------------------------------------------------------

#create and save LSTM models
for i in range (9):
    reset_globals()
    _ngrams_ = i+2
    start_s = time.time()
    lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, lstm_tokenizer = sequences_to_dl_ngrams(normal_data[normal_train])
    #lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, lstm_tokenizer = sequences_to_dl_ngrams(normal_data[0:1000])
    model = create_LSTM_model(lstm_ngrams_num, lstm_vocab_size, share_of_data=1)
    end_s = time.time()
    print("Lstm model create ", _ngrams_, " done in ", end_s - start_s)
    #Saving to disk
    save_string = "Loop_LSTM_"+"19012022_"+data+"_"+str(i)
    model.save(save_string)
    with open(save_string+"_tokenizer.pickle", 'wb') as handle:
        pickle.dump(lstm_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Load predict with models from disk
lstm_preds_all = []
for i in range (9):
    reset_globals()
    _ngrams_ = i+2
    start_s = time.time()

    #Load correct model
    save_string = "Loop_LSTM_"+"19012022_"+data+"_"+str(i)
    #model.save(save_string)
    model = keras.models.load_model(save_string)
    # saving tokenizer
    #with open(save_string+"_tokenizer.pickle", 'wb') as handle:
    #    pickle.dump(lstm_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #load tokenizer
    with open(save_string+"_tokenizer.pickle", 'rb') as handle:
        lstm_tokenizer = pickle.load(handle)

    lstm_preds = list()
    start_s = time.time()
    if (data=="pro"):
        for normal_s in normal_test:
            preds = give_preds_lstm(normal_s)
            lstm_preds.append(preds)
            #print(".")
        lstm_preds_all.append(lstm_preds)    
    elif(data=="hdfs"):
        lstm_preds_all.append(give_preds_lstm_2(normal_test))   
    end_s = time.time()
    print("ngrams infer ", _ngrams_, " done in ", end_s - start_s)

lstm_preds_all_mean = list()
for lstm_preds in lstm_preds_all:
    lstm_preds_means = list()
    if (data=="pro"):
        for lstm_pred in lstm_preds:
            lstm_preds_means.append(np.mean(lstm_pred))
        lstm_preds_all_mean.append(lstm_preds_means)
    elif(data=="hdfs"):
        lstm_preds_all_mean.append(np.mean(lstm_preds))

for i in range (9):
    print(np.mean(lstm_preds_all_mean[i]))




