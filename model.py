import tensorflow as tf


class MMOEDense(object):
    def __init__(self, nTasks, nExperts, inputDim, expertDim, hiddenDim, taskWeights, lr=0.01):
        self.nTasks = nTasks
        self.nExperts = nExperts
        self.inputDim = inputDim
        self.expertDim = expertDim
        # experts ops are independent, so don't use loop
        # experts (nExperts, inputDim, expertDim)
        expertInit = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
        self.experts = tf.Variable(expertInit(shape=(nExperts, inputDim, expertDim), dtype=tf.float32), name="experts")
        # gates, (nTasks, inputDim, nExpert)
        gateInit = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
        self.gates = tf.Variable(gateInit(shape=(nTasks, inputDim, nExperts), dtype=tf.float32), name="gates")
        # towers' mlp for each task, (nTasks, expertDim, hiddenDim)
        towersInit = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
        self.towers = tf.Variable(towersInit(shape=(nTasks, expertDim, hiddenDim), dtype=tf.float32), name="towers")
        # towers out
        towersOut = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
        self.outs = tf.Variable(towersOut(shape=(nTasks, hiddenDim, 1), dtype=tf.float32), name="outs")
        # target loss weights, (1, nTasks)
        self.tasksWeights = tf.constant([taskWeights])
        assert self.tasksWeights.shape[1] == nTasks
        self.trainableWeights = [
            self.experts,
            self.gates,
            self.towers,
            self.outs
        ]
        self.opt = tf.optimizers.Adam(learning_rate=lr)

    def __call__(self, input, labels=None):
        """
            @input: (batch, inputDim)
            @labels: (batch, nTasks)
        """
        # (batch, 1, 1, inputDim), second dim is for expert broadcast, third dim is for matmul
        input = tf.expand_dims(tf.expand_dims(input, axis=1), axis=1)

        # (batch, 1, 1, inputDim) X (nExperts, inputDim, expertDim) -> (batch, nExperts, 1, expertDim) -> (batch, 1, nExperts, expertDim)
        expertsOut = tf.transpose(tf.matmul(input, self.experts), [0, 2, 1, 3])

        # (batch, 1, 1, inputDim) X (nTasks, inputDim, nExperts) -> (batch, nTasks, 1, nExperts)
        # TODO: mask for seq padding
        weights = tf.nn.softmax(tf.matmul(input, self.gates), 3)
        # (batch, nTasks, 1, nExperts) X (batch, 1, nExperts, expertDim) -> (batch, nTasks, 1, expertDim)
        towersIn = tf.matmul(weights, expertsOut)
        # (batch, nTasks, 1, expertDim) X (nTasks, expertDim, hiddenDim) -> (batch, nTasks, 1, hiddenDim)
        towersHidden = tf.nn.relu(tf.matmul(towersIn, self.towers))
        # (batch, nTasks, 1, hiddenDim) X (nTasks, hiddenDim, 1) -> (batch, nTasks, 1, 1) -> (batch, nTasks)
        outs = tf.squeeze(tf.matmul(towersHidden, self.outs), axis=[2, 3])

        if labels is not None:
            # train
            # (batch, nTasks)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels, outs)
            # loss fusion, (1, nTasks) X (batch, nTasks, 1) -> (batch, 1, 1) -> (batch, )
            losses = tf.squeeze(tf.matmul(self.tasksWeights, tf.expand_dims(losses, axis=2)), axis=[1, 2])
            losses = tf.reduce_mean(losses)
            return losses
        else:
            # infer
            return tf.nn.sigmoid(outs)