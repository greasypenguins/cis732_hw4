#Resources used:
#Python docs
#sklearn docs
#https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
#https://en.wikipedia.org/wiki/Confusion_matrix
#https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/

import argparse
import random

class data:
    def __init__(self):
        #Initialize data structures
        self.attributes = list()
        self.data = list()
        self.data_verbose = list() #Can be derived from attributes and data

    def split_percent(self, percent):
        first_num = round(percent * len(self.data))
        return self.split_num(first_num)

    def split_num(self, first_num):
        first = data()
        second = data()

        first.attributes = self.attributes
        second.attributes = self.attributes

        first.data = self.data[:first_num]
        first.data_verbose = self.data_verbose[:first_num]
        second.data = self.data[first_num:]
        second.data_verbose = self.data_verbose[first_num:]

        return first, second

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

    def split_inputs_output(self):
        inputs = data()
        output = data()

        inputs.attributes = self.attributes[:-1]
        output.attributes = list()
        output.attributes.append(self.attributes[-1])

        inputs.data = [datapoint[:-1] for datapoint in self.data]
        inputs.data_verbose = [datapoint[:-1] for datapoint in self.data_verbose]
        
        for datapoint in self.data:
            new_datapoint = list()
            new_datapoint.append(datapoint[-1])
            output.data.append(new_datapoint)

        for datapoint in self.data_verbose:
            new_datapoint = list()
            new_datapoint.append(datapoint[-1])
            output.data_verbose.append(new_datapoint)

        return inputs, output

class arff(data):
    def __init__(self, f):
        #Initialize data
        data.__init__(self)

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

            if at_type == "RELATION":
                continue

            elif at_type == "ATTRIBUTE":
                attribute = list()
                attribute.append(words[1])
                attribute.append(words[2:])
                self.attributes.append(attribute)

            elif at_type == "DATA":
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

            else:
                raise Exception("Unknown arff format")

class naive_bayes:
    def __init__(self, dataset):
        self.dataset = dataset
        self.inputs = None
        self.output = None

    def predict(self, input_datapoint):
        #For each label, predict probability of label
        probs = list()
        for i, _ in enumerate(self.dataset.attributes[-1][1]):
            datapoint = input_datapoint[:]
            datapoint.append(i)
            probs.append(self.probability_given_inputs(datapoint))

        #Choose highest probability and return that class (MAP)
        max_i = 0.0
        max_prob = 0.0
        for i, prob in enumerate(probs):
            if prob > max_prob:
                max_i = i
                max_prob = prob
        
        return max_i

    def probability_of_value(self, position, value):
        count = 0.0
        for datapoint in self.dataset.data:
            if datapoint[position] == value:
                count += 1.0
        return count / float(len(self.dataset))

    def probability_of_value_given_value(self, pos, val, given_pos, given_val):
        count = 0.0
        total = 0.0
        for datapoint in self.dataset.data:
            if datapoint[given_pos] == given_val:
                total += 1.0
                if datapoint[pos] == val:
                    count += 1.0
        if total == 0:
            raise Exception("Value {} never found in position {}".format(given_val, given_pos))
        return count / total

    def probability_given_inputs(self, datapoint):
        #P(y | x1, x2, ... xn) = P(x1|y) * P(x2|y) * … P(xn|y) * P(y)
        prob = 1.0
        
        for pos, val in enumerate(datapoint[:-1]):
            prob *= self.probability_of_value_given_value(pos, val, -1, datapoint[-1])
        
        prob *= self.probability_of_value(-1, datapoint[-1])

        return prob

def main():
    #Get arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--filename", default="titanic.arff", help="name of .arff file, if different from titanic.arff")
    args = ap.parse_args()

    #Parse ARFF file
    with open(args.filename) as f:
        arff_data = arff(f)

    #Prepare data
    arff_data.shuffle()

    training_data, validation_data = arff_data.split_percent(.75)
    #print(training_data)
    #print(validation_data)

    training_inputs, training_output = training_data.split_inputs_output()
    validation_inputs, validation_output = validation_data.split_inputs_output()
    #print(validation_inputs)
    #print(validation_output)

    #Perform Naive Bayes
    nb = naive_bayes(training_data)

    datapoint = [0, 0, 0, 0]

    prob = nb.probability_given_inputs(datapoint)
    print("Probability of {} given {} is {:.2f}%".format(datapoint[-1], datapoint[:-1], 100.0 * prob))

    pred = nb.predict(datapoint[:-1])
    print("Prediction of {} is {}".format(datapoint[:-1], pred))



    return

if __name__ == "__main__":
    main()