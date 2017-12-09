sess = tf.Session()

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def join_cifar_data(dict_1, dict_2):
    dict_1['data'] = np.concatenate((dict_1['data'], dict_2['data']), axis=0)
    dict_1['labels'] = dict_1['labels'] + dict_2['labels']
    return dict_1

train_set = []
for i in range(1,6): # run 1 to 5
    temp = unpickle(os.path.join('cifar-10-batches-py', 'data_batch_' + str(i)))
    data = {'data': temp['data'], 'labels': sess.run(tf.one_hot(temp['labels'], 10))}
    train_set.append(data)
    
temp = unpickle(os.path.join('cifar-10-batches-py', 'test_batch'))
test_set = {'data': temp['data'], 'labels': sess.run(tf.one_hot(temp['labels'], 10))}