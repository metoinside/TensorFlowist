import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

#Loading the Boston Housing prices from SciKit-Learn Dataset
total_features, total_prices = load_boston(True)

#Split dataset into 3 groups: Train, Validate and Test
train_feature = scale(total_features[:300])
train_price = total_prices[:300]

validation_feature = scale(total_features[300:400])
validation_price = total_prices[300:400]

test_feature = scale(total_features[400:])
test_prices = total_prices[400:]

#Create tensorflow variables for y=w.x+b model
w = tf.Variable(tf.truncated_normal([13,1], mean=0.0, stddev=1.0, dtype=tf.float64))
b = tf.Variable(tf.zeros(1, dtype=tf.float64))

#Simply make the matrix calculation and error
def calc(x, y):
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y-predictions))

    return [predictions, error]

y, cost = calc(train_feature, train_price)
learning_rate = 0.025
epochs = 3000

points = [[], []]

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


## Start Tensorflow Sessions after the setup

with tf.Session() as Sess:
    Sess.run(init)
    for i in list(range(epochs)):
        Sess.run(optimizer)

        if i%10 == 0:
            points[0].append(i+1)
            points[1].append(Sess.run(cost))

        if i%100 == 0:
            print Sess.run(cost)

    valid_cost = calc(validation_feature, validation_price)[1]
    print('Validation error: ', Sess.run(valid_cost), '\n')

    test_cost = calc(test_feature, test_prices)[1]
    print('Validation error: ', Sess.run(test_cost), '\n')

