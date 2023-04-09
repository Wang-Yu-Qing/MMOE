import tensorflow as tf
from utils import *
from model import MMOEDense


if __name__ == '__main__':
    args = parseArgs()
    if (args.prepareRecords): prepareDataset(args)

    trainData = tf.data.TFRecordDataset(args.trainRecordPath).map(parseSample).batch(args.trainBatchSize)
    taskWeights = [float(x) for x in args.taskWeights.split(',')]
    model = MMOEDense(args.nTasks, args.nExperts, args.inputDim, args.expertDim, args.hiddenDim, taskWeights)
    inputBN = tf.keras.layers.BatchNormalization(axis=1)
    print("start training")

    for epoch in range(args.epochs):
        epochTotalLoss = 0
        for step, (fea, marital, income) in enumerate(trainData):
            labels = tf.cast(tf.transpose(tf.stack([marital, income]), [1, 0]), tf.float32)
            # batch norm for census features:
            fea = inputBN(fea, training=True)
            with tf.GradientTape() as tape:
                losses = model(fea, labels)
                grads = tape.gradient(losses, model.trainableWeights)
                model.opt.apply_gradients(zip(grads, model.trainableWeights))
            
            epochTotalLoss += losses
            epochAvgLoss = epochTotalLoss / (step + 1)

            if ((step + 1) % args.printSteps == 0):
                print("| epoch: {:03d} | step: {:06d} | epoch avg loss: {:.4f}".format(epoch, step + 1, epochAvgLoss))

    # test
    from sklearn.metrics import roc_auc_score
    testData = tf.data.TFRecordDataset(args.testRecordPath).map(parseSample).batch(args.testBatchSize)

    maritalLogits, maritalLabels = [], []
    incomeLogits, incomeLabels = [], []
    for step, (fea, marital, income) in enumerate(testData):
        labels = tf.cast(tf.transpose(tf.stack([marital, income]), [1, 0]), tf.float32)
        # batch norm for census features:
        fea = inputBN(fea, training=False)
        logits = model(fea) # (batch, nTasks)
        for logit, label in zip(logits, labels):
            maritalLogits.append(logit[0])
            maritalLabels.append(label[0])
            incomeLogits.append(logit[1])
            incomeLabels.append(label[1])

    maritalAuc = roc_auc_score(maritalLabels, maritalLogits)
    incomeAuc = roc_auc_score(incomeLabels, incomeLogits)

    print("maritalAuc: {:.04f}, incomeAuc: {:0.4f}".format(maritalAuc, incomeAuc))