#Resources used:
#Python docs
#sklearn docs
#https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
#https://en.wikipedia.org/wiki/Confusion_matrix

import argparse
import random

class data:
    def __init__(self):
        #Initialize data structures
        self.attributes = list()
        self.data = list()
        self.data_verbose = list()

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

    #Print for debug purposes
    print(training_data)
    print(validation_data)

    #Perform Naive Bayes

    return

if __name__ == "__main__":
    main()