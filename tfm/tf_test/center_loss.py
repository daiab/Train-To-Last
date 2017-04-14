import tensorflow as tf
import numpy as np

def get_center_loss(features, labels, alpha, num_classes):
    # alpha:中心的更新比例
    # 获取特征长度
    len_features = features.get_shape()[1]
    # 建立一个变量，存储每一类的中心，不训练9
    centers = tf.get_variable("center", [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    # 将特征reshape成一维
    labels = tf.reshape(labels, [-1])

    # 获取当前batch每个样本对应的中心
    centers_batch = tf.gather(centers, labels)
    # 计算center loss的数值
    loss = tf.nn.l2_loss(features - centers_batch)

    # 以下为更新中心的步骤
    diff = centers_batch - features

    # 获取一个batch中同一样本出现的次数，这里需要理解论文中的更新公式
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    # 更新中心
    centers = tf.scatter_sub(centers, labels, diff)

    return loss, centers

# ================================================

nclass=10
ndims=5
def get_embed_centers(feature, label):
    ''' Exponential moving window average. Increase decay for longer windows [0.0 1.0]'''
    decay = 0.95
    with tf.variable_scope('embed', reuse=True):
        embed_ctrs = tf.get_variable("ctrs")

    label = tf.reshape(label, [-1])
    old_embed_ctrs_batch = tf.gather(embed_ctrs, label)
    dif = (1 - decay) * (old_embed_ctrs_batch - feature)
    embed_ctrs = tf.scatter_sub(embed_ctrs, label, dif)
    embed_ctrs_batch = tf.gather(embed_ctrs, label)
    return embed_ctrs_batch


with tf.Session() as sess:
    with tf.variable_scope('embed'):
        embed_ctrs = tf.get_variable("ctrs", [nclass, ndims], dtype=tf.float32,
                        initializer=tf.constant_initializer(0), trainable=False)
    label_batch_ph = tf.placeholder(tf.int32)
    embed_batch_ph = tf.placeholder(tf.float32)
    embed_ctrs_batch = get_embed_centers(embed_batch_ph, label_batch_ph)
    sess.run(tf.initialize_all_variables())
    tf.get_default_graph().finalize()

# ================================================

ndims = 2
nclass = 4
nbatch = 100

with tf.variable_scope('center'):
    center_sums = tf.get_variable("sums", [nclass, ndims], dtype=tf.float32,
                    initializer=tf.constant_initializer(0), trainable=False)
    center_cts = tf.get_variable("cts", [nclass], dtype=tf.float32,
                    initializer=tf.constant_initializer(0), trainable=False)

def get_new_centers(embeddings, indices):
    '''
    Update embedding for selected class indices and return the new average embeddings.
    Only the newly-updated average embeddings are returned corresponding to
    the indices (including duplicates).
    '''
    with tf.variable_scope('center', reuse=True):
        center_sums = tf.get_variable("sums")
        center_cts = tf.get_variable("cts")

    # update embedding sums, cts
    if embeddings is not None:
        ones = tf.ones_like(indices, tf.float32)
        center_sums = tf.scatter_add(center_sums, indices, embeddings, name='sa1')
        center_cts = tf.scatter_add(center_cts, indices, ones, name='sa2')

    # return updated centers
    num = tf.gather(center_sums, indices)
    denom = tf.reshape(tf.gather(center_cts, indices), [-1, 1])
    return tf.div(num, denom)


with tf.Session() as sess:
    labels_ph = tf.placeholder(tf.int32)
    embeddings_ph = tf.placeholder(tf.float32)

    unq_labels, ul_idxs = tf.unique(labels_ph)
    indices = tf.gather(unq_labels, ul_idxs)
    new_centers_with_update = get_new_centers(embeddings_ph, indices)
    new_centers = get_new_centers(None, indices)

    sess.run(tf.initialize_all_variables())
    tf.get_default_graph().finalize()

    for i in range(100001):
        embeddings = 100*np.random.randn(nbatch, ndims)
        labels = np.random.randint(0, nclass, nbatch)
        feed_dict = {embeddings_ph:embeddings, labels_ph:labels}
        rval = sess.run([new_centers_with_update], feed_dict)
        if i % 1000 == 0:
            feed_dict = {labels_ph:range(nclass)}
            rval = sess.run(new_centers, feed_dict)
            print('\nFor step ', i)
            for iclass in range(nclass):
                print('Class %d, center: %s' % (iclass, str(rval[iclass])))