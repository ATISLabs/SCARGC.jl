using Pkg

Pkg.add("SCARGC")

using SCARGC

data = extractValuesFromFile("../src/datasets/synthetic/1CDT.txt", 16000, 3)

predictedLabels, _ = scargc_1NN(data, 5.0, 300, 2)

predictedLabels, accuracy = scargc_1NN(data, 5.0, 300, 2)

println("Predicted labels: ", predictedLabels)
println("Accuracy: ", accuracy, "%")
