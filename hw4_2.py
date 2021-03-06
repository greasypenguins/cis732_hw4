#Resources used:
#Python docs
#sklearn docs
#https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
#https://en.wikipedia.org/wiki/Confusion_matrix
#https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/

import argparse
import random
from sklearn.neural_network import MLPClassifier
import time

class data:
    def __init__(self):
        #Initialize data structures. Do not edit these from outside the class!
        self.num_training = None
        self.attributes = list()
        self.data = list()
        self.data_verbose = list() #Can be derived from attributes and data

        return

    def split_percent(self, percent):
        num_first = round(percent * len(self.data))
        self.split_num(num_first)
        return

    def split_num(self, num_training):
        self.num_training = num_training
        return

    def __len__(self):
        return len(self.data)

    def __str__(self):
        a = ""
        for attribute in self.attributes:
            a = "{}  {}\n".format(a, attribute[0])
            for label in attribute[1]:
                a = "{}    {}\n".format(a, label)

        d = ""
        for i, datapoint in enumerate(self.data):
            d = "{}  {}:".format(d, i)
            for num_label in datapoint:
                d = "{} {}".format(d, num_label)
            d = "{}\n".format(d)

        final = "Length: {}\n\nAttributes:\n{}\nData:\n{}".format(len(self), a, d)

        return final

    def shuffle(self):
        zipped = [x for x in zip(self.data, self.data_verbose)]
        random.shuffle(zipped)
        self.data = [x[0] for x in zipped]
        self.data_verbose = [x[1] for x in zipped]

        return

    def get_training_datapoints(self):
        return self.data[:self.num_training]

    def get_validation_datapoints(self):
        return self.data[self.num_training:]

    def get_attribute_name(self, pos):
        return self.attributes[pos][0]

    def get_attribute_label_name(self, pos, val):
        return self.attributes[pos][1][val]

class arff(data):
    def __init__(self, f):
        #Initialize data
        data.__init__(self)
        self.read_data(f, True)

    def read_data(self, f, first_time):
        #While loop reading lines for @s
        at_type = None
        for line in f:
            line = line.replace(",", " ").replace("{", "").replace("}", "").replace("\"", "")

            words = line.split()

            #If this line is a new at type
            if line[0] == "@":
                at_type = words[0][1:]

                if at_type == "DATA":
                    continue

            if at_type == "DATA":
                if len(words) != len(self.attributes):
                    raise Exception("Datapoint does not have the correct number of attributes")
                datapoint = list()
                for i, word in enumerate(words):
                    if word not in self.attributes[i][1]:
                        raise Exception("Unknown attribute encountered")
                    for j, attribute_label in enumerate(self.attributes[i][1]):
                        if word == attribute_label:
                            datapoint.append(j)
                self.data_verbose.append(words)
                self.data.append(datapoint)

            elif not first_time:
                #All non-data fields were already read previously, so no need to process them
                continue

            elif at_type == "RELATION":
                #This seems useless?
                continue

            elif at_type == "ATTRIBUTE":
                attribute = list()
                attribute.append(words[1])
                attribute.append(words[2:])
                self.attributes.append(attribute)

            else:
                raise Exception("Unknown arff format")

        return

    def read_val_data(self, f):
        self.num_training = len(self.data)
        self.read_data(f, False)

        return

def accuracy(pred_outputs, actual_outputs):
    if len(pred_outputs) != len(actual_outputs):
        raise Exception("Different length lists")
    num_correct = 0.0
    for i in range(len(pred_outputs)):
        if pred_outputs[i] == actual_outputs[i]:
            num_correct += 1.0
    return num_correct / float(len(pred_outputs))

def get_confusion_matrix(num_possible_vals, pred_vals, actual_vals):
    if len(pred_vals) != len(actual_vals):
        raise Exception("Invalid length of input or incorrect position")
    cm = list()
    for row_val in range(num_possible_vals):
        row = list()
        for col_val in range(num_possible_vals):
            count = 0
            #Find number of times that row_val appeared in pred_vals at the same index that col_val appeared in actual_vals
            for i, pred_val in enumerate(pred_vals):
                if (pred_val == row_val) and (actual_vals[i] == col_val):
                    count += 1
            row.append(count)
        cm.append(row)
    return cm

def main():
    print("WARNING: This script assumes the prediction/output attribute is the last one for each data point in the .arff file!")
    #Get arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename",
        default="titanic.arff",
        help="Name of .arff file, if different from titanic.arff")
    ap.add_argument("-l", "--hidden_layers",
        default="1",
        help="Number if hidden layers in the Multilayer Perceptron")
    ap.add_argument("-s", "--split",
        default="0.75",
        help="Percent of data used for training")
    ap.add_argument("-v", "--val_filename",
        default=None,
        help="Optional name of valifation .arff file; if specified, the split argument is ignored")
    args = ap.parse_args()

    #Parse ARFF file
    print("Reading \"{}\"".format(args.filename))
    with open(args.filename) as f:
        arff_data = arff(f)

    if args.val_filename is None:
        #Prepare data
        print("Shuffling data")
        arff_data.shuffle()
        split_percent = float(args.split)
        print("Splitting data into {:.2f}% training and {:.2f}% validation".format(100.0 * split_percent, 100.0 * (1.0 - split_percent)))
        arff_data.split_percent(split_percent)
    else:
        #Parse validation ARFF file
        print("Reading \"{}\" (validation data)".format(args.val_filename))
        print("  Assuming only data differs from training .arff file")
        with open(args.val_filename) as f:
            arff_data.read_val_data(f)
        split_percent = float(arff_data.num_training) / float(len(arff_data))
        print("Split is {:.2f}% training and {:.2f}% validation".format(100.0 * split_percent, 100.0 * (1.0 - split_percent)))

    #Break up data
    training_datapoints = arff_data.get_training_datapoints()
    validation_datapoints = arff_data.get_validation_datapoints()

    training_inputs = [datapoint[:-1] for datapoint in training_datapoints]
    training_outputs = [datapoint[-1] for datapoint in training_datapoints]
    validation_inputs = [datapoint[:-1] for datapoint in validation_datapoints]
    validation_outputs = [datapoint[-1] for datapoint in validation_datapoints]

    #Make Multilayer Perceptron
    print("Using Multilayer Perceptron")
    mlp = MLPClassifier(activation="logistic")
    mlp.n_layers = int(args.hidden_layers)

    print("Fitting model")
    start_time = time.time()
    mlp.fit(training_inputs, training_outputs)
    end_time = time.time()
    print("  Time taken: {:.2f} s".format(end_time - start_time))

    #Demo a data point
    print("Predicting a sample data point")
    datapoint = list()
    for _ in range(len(arff_data.attributes)):
        datapoint.append(0)

    datapoint = [0, 0, 0, 0]
    datapoint_verbose = [arff_data.get_attribute_label_name(pos, val) for pos, val in enumerate(datapoint)]

    x = list()
    x.append(datapoint[:-1])

    pred = mlp.predict(x)[0]
    pred_verbose = arff_data.get_attribute_label_name(len(datapoint) - 1, pred)
    print("  Prediction of {} is {}".format(datapoint[:-1], pred))
    print("  Prediction of {} is {}".format(datapoint_verbose[:-1], pred_verbose))

    #Run Multilayer Perceptron predictions for training and validation data
    print("Predicting training outputs")
    start_time = time.time()
    pred_training_outputs = mlp.predict(training_inputs)
    end_time = time.time()
    print("  Time taken: {:.2f} s".format(end_time - start_time))

    print("Predicting validation outputs")
    start_time = time.time()
    pred_validation_outputs = mlp.predict(validation_inputs)
    end_time = time.time()
    print("  Time taken: {:.2f} s".format(end_time - start_time))

    training_error = 1.0 - accuracy(pred_training_outputs, training_outputs)
    validation_error = 1.0 - accuracy(pred_validation_outputs, validation_outputs)

    print("Training error: {:.2f}%".format(100.0 * training_error))
    print("Validation error: {:.2f}%".format(100.0 * validation_error))

    #Generate confusion matrix
    print("Generating confusion matrices")
    num_possible_vals = len(arff_data.attributes[-1][1])
    training_cm = get_confusion_matrix(num_possible_vals, pred_training_outputs, training_outputs)
    validation_cm = get_confusion_matrix(num_possible_vals, pred_validation_outputs, validation_outputs)

    print("Confusion matrix format:")
    print("           Act 0 | Act 1 | Act 2 | ...")
    print("  Pred 0 |   x   |   x   |   x   | ...")
    print("  Pred 1 |   x   |   x   |   x   | ...")
    print("  Pred 2 |   x   |   x   |   x   | ...")
    print("  ...    |  ...  |  ...  |  ...  | ...")

    print("Training confusion matrix:")
    for row in training_cm:
        print("  {}".format(row))

    print("Validation confusion matrix:")
    for row in validation_cm:
        print("  {}".format(row))

    return

if __name__ == "__main__":
    main()
