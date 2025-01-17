"""
Main module for `SCARGC.jl`, the Julia implementation of SCARGC algorithm.

From this module, two functions are exported for public use:

- [`scargc_1NN`](@ref). The 1-Nearest Neighbor SCARGC implementation.
- [`extractValuesFromFile`](@ref). A functio to extract the data values from a dataset file.
"""
module SCARGC

using Distances, Statistics
using PyCall

export scargc_1NN, extractValuesFromFile

"""
    scargc_1NN(
        dataset         -> "dataset used in the algorighm"
        percentTraining -> "amount of data that is goung to be used as training data"
        maxPoolSize     -> "maximum instances that the pool size can store"
        K               -> "number of clusters"
    )

SCARGC implementation with the Nearest Neighbor classifier for label predicting.
The function prints the final accuracy and returns the vector with all predicted labels and the final accuracy.

The function starts getting the labeled and unlabeled and, with them, it creates the initial centroids.
Then, a loop starts over the unlabeled data, storing the instance and the predicted label (predicted with the
classification model). 

When the pool reaches the maximum, represented by `maxPoolSize`, a clustering step is
made on the data stored in the pool to get the centroids from the current iteration (represented as `tempCurrentCentroids`
before receiving the labels and `currentCentroids` after it). With the centroids from the past iteration (represented 
as `centroids`), we find the current iteration's centroids' labels and create the intermediary centroids, represented
as `intermed`, that stores the median between the past and current centroids and the current iretation's
centroid's labels (the reason to store that is to store the drift between the past and the current iteration).

After getting the labels, the past iteration's centroids receive the values stored in `intermed` and a new labels are
found using both centroids from current and past iterations. These labels are going to be useful to get the concordance
between them and the labels stored in the pool to know if the model is going to be updated.
The last part, after calculating the concordance, is update the classification model if the concordance is different of 
100%.

For example, let's predict some data in the **1CHT** dataset.

We start loading the dataset and reading the values from it.

```@example
using SCARGC

# loading dataset
dataset = "datasets/synthetic/1CHT.txt"

# extracting values from the dataset file and storing into `data`
data = extractValuesFromFile(dataset, 16000, 3)

# printing the read dataset
prinln(data)

# predicting labels and storing them in `predictedLabels` and the accuracy in `accuracy`
predictedLabels, accuracy = scargc_1NN(data, 0.3125, 300, 2)
```

If we print the results, we're gonna have

```julia
# predicted labels
11000-element Array{Int64,1}:
 1
 2
 1
 1
 1
 2
 1
 1
 ⋮
 2
 1
 2
 1
 2
 2
 2

# accuracy ~
99.73040752351096
```

The accuracy happens if you're using a dataset with the actual labels of all instances. Then
the function can compare the prediction with the actual label to get the accuracy.
"""
function scargc_1NN(dataset::Array{T, 2} where {T<:Number}, percentTraining::Float64, maxPoolSize::Int64, K::Int64)
    labeledData, labeledDataLabels, streamData, streamLabels, features = fitData(dataset, percentTraining)
    centroids = findCentroids(features, labeledData, labeledDataLabels, K)

    predictedLabels = zeros(length(streamLabels))

    accuracyVector = zeros(size(streamData, 1))
    poolData = zeros(maxPoolSize, size(streamData, 2) + 1)
    poolDataIndex = 1

    for stream = 1:size(streamData, 1)
        testInstance = streamData[stream, :]
        currentLabel = streamLabels[stream]

        predictedLabel, _ = knnClassification(labeledData, labeledDataLabels, testInstance)

        poolData[poolDataIndex, 1:end-1] = testInstance
        poolData[poolDataIndex, end] = predictedLabel
        poolDataIndex += 1

        if poolDataIndex > maxPoolSize
            py"""
            from sklearn.cluster import KMeans
            import numpy as np
            import sys, os
            import warnings

            warnings.filterwarnings("ignore")

            def kmeans(X, k, init, centroids=[]):
                if (init == 0):
                    return KMeans(n_clusters=k).fit(X).cluster_centers_
                else:
                    centroids = np.array(centroids)
                    sys.stdout = open(os.devnull, 'w')
                    return KMeans(n_clusters=k, init=centroids).fit(X).cluster_centers_"""

            tempCurrentCentroids = py"kmeans"(poolData[:, 1:end-1], K, 1, centroids[end-K+1:end, 1:end-1])

            intermed, centroidLabels = findLabelForCurrentCentroids(centroids, tempCurrentCentroids)

            currentCentroids = zeros(size(tempCurrentCentroids, 1), size(tempCurrentCentroids, 2) + 1)
            currentCentroids[:, 1:end-1] = tempCurrentCentroids
            currentCentroids[:, end] = centroidLabels

            centroids = intermed

            newLabeledData = calculateNewLabeledData(poolData, currentCentroids, centroids)
            
            concordantLabelCount = count(label -> label == 1, poolData[:, end] .== newLabeledData)

            centroids, labeledData, labeledDataLabels = updateInformation(poolData, centroids, labeledData, labeledDataLabels,
                                                                        newLabeledData, currentCentroids, intermed, 
                                                                        concordantLabelCount, maxPoolSize)

            poolData = zeros(maxPoolSize, size(streamData, 2) + 1)
            poolDataIndex = 1
        end
        
        predictedLabels[stream] = predictedLabel

        if predictedLabel == currentLabel
            global accuracyVector[stream] = 1
        end
    end

    finalAccuracy = (sum(accuracyVector)/size(streamData, 1)) * 100

    #println(finalAccuracy)

    return predictedLabels, finalAccuracy
end

"""
    extractValuesFromFile(
        fileName -> "name of the file that you're trying to read"
        rows     -> "number of rows in the file"
        columns  -> "number of columns in the file"
    )

The function reads a file and creates a matrix with the file's values.
Using the function `readuntil` and `parse` we can converte each value, separeted by comma.

The function returns a matrix with the values in the file.

For example, let's suppose we have a text file, called "example.txt", with **10** lines and **7** columns.

```
2.91120, 2.94941, 2.92468, 0.88324, 6.00512, 5.88021, 3
4.97443, 2.31345, 5.21826, 8.13938, 7.47390, 8.24444, 1
3.13772, 7.87192, 8.35722, 7.41002, 6.95894, 5.20002, 2
3.74301, 6.39082, 3.89166, 8.76716, 1.66374, 6.75246, 3
8.98766, 4.42780, 4.94278, 7.05217, 5.21106, 3.69790, 1
4.38337, 4.91497, 6.16593, 8.73148, 6.17557, 0.90185, 2
0.97263, 2.70122, 0.00343, 7.20105, 1.63296, 8.26112, 1
0.50329, 3.54209, 8.47787, 2.59826, 3.93825, 2.58200, 3
6.75223, 4.59601, 2.02472, 8.10523, 3.65602, 6.91874, 1
8.98924, 2.33177, 8.34892, 4.03544, 8.57646, 7.93690, 1
```

We're gonna run this function with the following values:

- `fileName` is going to receive the path to the file. In this case, it's just "example.txt";
- `rows` is going to receive 10 and
- `columns` is going to receive 7

Then, the function starts and reads each value separated by "," and stores these values into 
a matrix. This matrix is returned in the end of the function, in the format:

```julia
10×7 Array{Float64,2}:
2.91120, 2.94941, 2.92468, 0.88324, 6.00512, 5.88021, 3
4.97443, 2.31345, 5.21826, 8.13938, 7.47390, 8.24444, 1
3.13772, 7.87192, 8.35722, 7.41002, 6.95894, 5.20002, 2
3.74301, 6.39082, 3.89166, 8.76716, 1.66374, 6.75246, 3
8.98766, 4.42780, 4.94278, 7.05217, 5.21106, 3.69790, 1
4.38337, 4.91497, 6.16593, 8.73148, 6.17557, 0.90185, 2
0.97263, 2.70122, 0.00343, 7.20105, 1.63296, 8.26112, 1
0.50329, 3.54209, 8.47787, 2.59826, 3.93825, 2.58200, 3
6.75223, 4.59601, 2.02472, 8.10523, 3.65602, 6.91874, 1
8.98924, 2.33177, 8.34892, 4.03544, 8.57646, 7.93690, 1
```

This matrix is then ready to be used along the code.

"""
function extractValuesFromFile(fileName::String, rows::Int64, columns::Int64)
    file = open(fileName)

    fileValues = zeros(rows, columns)

    for row = 1:rows
        for column = 1:columns-1
            global fileValues[row, column] = parse(Float64, readuntil(file, ','))
        end

        global fileValues[row, columns] = parse(Float64, readuntil(file, '\n'))
    end

    close(file)

    return fileValues
end

"""
    fitData(
        dataset         -> "the whole dataset loaded from file system"
        percentTraining -> "percentage of the amount of rows to be used as labeled data"
    )

Divides the dataset into labeled data and unlabeled data, training and testing data respectively.
These arrays are gonna be used in the whole program, so the fitting part is considered extremelly important because, if 
something's wrong in the this step, the whole application is gonna fail.

The return of this function has the following variables:
    - labeledData       -> "`percentTraining`% of the dataset features"
    - labeledDataLabels -> "also `percentTraining`% of the dataset labels"
    - streamData        -> "remaining data of the dataset"
    - streamLabels      -> "remaining labels of the dataset"
    - features          -> "number of features present in the data"
"""
function fitData(dataset::Array{T, 2} where {T<:Number}, percentTraining::Float64)
    rows = size(dataset, 1)
    columns = size(dataset, 2)

    labeledRowCount = Int(ceil((percentTraining/100) * rows))

    labeledData = dataset[1:labeledRowCount, 1:end-1]
    labeledDataLabels = dataset[1:labeledRowCount, end]

    streamData = dataset[labeledRowCount + 1:end, 1:end-1]
    streamLabels = dataset[labeledRowCount + 1:end, end]

    features = columns - 1

    return labeledData, labeledDataLabels, streamData, streamLabels, features
end

"""
    knnClassification(
        labeledData  -> "labeled data used to apply the euclidean distance"
        labels       -> "all data labels"
        testInstance -> "the instance we want to predict the output based on the nearest neighbor."
    )

Applies the Nearest Neighbor to the `testInstance`.
The function calculates, for each `labeledData`, the Euclidean Distance between it and the test instance and. 
If it's value is smaller then the smallest distance, the smallest distance recieves this distance value and the label 
value, at this index, is stored.
The return of the function is the output and the data from the nearest neighbor.
"""
function knnClassification(labeledData::Array{T, 2} where {T<:Number}, labels::Array{T} where {T<:Number}, 
                           testInstance::Array{T} where {T<:Number})
    output = nothing
    resultData = nothing
    smallerDistance = Inf

    for row = 1:size(labeledData, 1)
        data = labeledData[row, :]

        distance = Distances.euclidean(testInstance, data)

        if (distance < smallerDistance)
            smallerDistance = distance
            output = labels[row]
            resultData = data
        end
    end

    return output, resultData
end

"""
    findCentroids(
        features          -> "number of features"
        labeledData       -> "labeled data used in classification"
        labeledDataLabels -> "labels of the labeled data used in classification"
        K                 -> "number of clusters"
    )

Finds the initial centroids used in program, using KMeans if necessary, with the data already fit. 
The function has two possible ways:
    - The value of K is the same as the number of classes; and
    - The value of K is different as the number of classes.

If the values are the same, the function creates the centroids as the median of the data. In other words, if these 
numbers are equal, the centroids have the value of the median of each feature.
Otherwise, if the values aren't the same, the KMeans method is used to define the initial centroids.

The function returns the centroids array.
"""
function findCentroids(features::Int64, labeledData::Array{T, 2} where {T<:Number}, 
                       labeledDataLabels::Array{T} where {T<:Number}, K::Int64)
    classes = unique(labeledDataLabels)
    totalClasses = length(classes)

    # Finding the first centroids
    # If the number of classes is the same as K, the initial centroid of each class is the mean of each feature.
    # If it's not, a clustering is necessary to find the centroids.
    if K == totalClasses
        centroids = zeros(totalClasses, features + 1)

        for class = 1:totalClasses
            tempCentroids = zeros(features)

            for feature = 1:features
                tempCentroids[feature] = median(labeledData[findall(label -> label == classes[class], 
                                                labeledDataLabels), feature])
            end
            
            centroids[class, 1:end-1] = tempCentroids
        end

        centroids[:, end] = classes
    else
        py"""
            from sklearn.cluster import KMeans
            import numpy as np
            import sys, os
            import warnings

            warnings.filterwarnings("ignore")

            def kmeans(X, k, init, centroids=[]):
                if (init == 0):
                    return KMeans(n_clusters=k).fit(X).cluster_centers_
                else:
                    centroids = np.array(centroids)
                    return KMeans(n_clusters=k, init=centroids).fit(X).cluster_centers_"""

        tempCentroids = py"kmeans"(labeledData, K, 0)

        centroidLabels = zeros(size(tempCentroids, 1))

        for row = 1:size(tempCentroids, 1)
            output, _ = knnClassification(labeledData, labeledDataLabels, tempCentroids[row, :])
            centroidLabels[row] = output
        end

        centroids = zeros(size(tempCentroids, 1), size(tempCentroids, 2) + 1)
        centroids[:, 1:end-1] = tempCentroids
        centroids[:, end] = centroidLabels
    end

    return centroids
end

"""
    findLabelForCurrentCentroids(
        pastCentroids    -> "centroids from previous iteration"
        currentCentroids -> "centroids from current iteration"
    )

The function uses the last iteration centroids' labels to define the current iteration centroids' labels.
Given the current centroids from the most recent unlabeled clusters and the past centroids from the previously labeled
clusters, each centroid from the past iteration have a label yi and each centroid from current iteration needs a label 
ˆyi. This label is obtained by aplying nearest neighbor algorithm. In other words, the label given to a centroid in the
current iteration is the same as the label given to it's nearest neighbor in the past iteration.

The function returns two arrays:
    intermed       -> matrix storing the median between the nearest neighbor's data and the current centroids and it's labels
    centroidLabels -> labels got for the current centroids
"""
function findLabelForCurrentCentroids(pastCentroids::Array{T, 2} where {T<:Number}, currentCentroids::Array{T, 2} where {T<:Number})
    intermed = zeros(size(currentCentroids, 1), size(pastCentroids, 2))
    centroidLabels = zeros(size(currentCentroids, 1))

    for row = 1:size(currentCentroids, 1)
        label, nearestData = knnClassification(pastCentroids[:, 1:end-1], pastCentroids[:, end], currentCentroids[row, :])

        medianMatrix = zeros(2, size(currentCentroids, 2))
        medianMatrix[1, :] = nearestData
        medianMatrix[2, :] = currentCentroids[row, :]

        intermed[row, 1:end-1] = median(medianMatrix, dims=1)
        intermed[row, end] = label

        centroidLabels[row] = label
    end

    return intermed, centroidLabels
end

"""
    calculateNewLabeledData(
        poolData          -> "data stored in the pool"
        currentCentroids  -> "centroids from current iteration"
        pastCentroids     -> "matrix with centroids and their labels"
    )

Given the updated centroid values and the pool data, the function calculates the new labeled data using nearest neighbor 
over the vcat of current centroids and the intermed centroids and the pool data. 
The result is going to have the labels got with new centroid values, from the updated classifier.
"""
function calculateNewLabeledData(poolData::Array{T, 2} where {T<:Number}, currentCentroids::Array{T, 2} where {T<:Number}, 
                                 pastCentroids::Array{T, 2} where {T<:Number})
    newLabeledData = zeros(size(poolData, 1))

    lblData = vcat(currentCentroids[:, 1:end-1], pastCentroids[:, 1:end-1])
    lblLabels = vcat(currentCentroids[:, end], pastCentroids[:, end])

    for row = 1:size(poolData, 1)
        output, _ = knnClassification(lblData, lblLabels, poolData[row, 1:end-1])

        newLabeledData[row] = output
    end

    return newLabeledData
end

"""
    updateInformation(
        poolData             -> "test instances with it's labels"
        centroids            -> "centroids from past iteration"
        labeledData          -> "labeled data used in classification"
        labeledDataLabels    -> "labels from the data in `labeledData`"
        newLabeledData       -> "data calculated with the new centroid values"
        currentCentroids     -> "centroids from the current iteration"
        intermed             -> "matrix with all median data between past and current iteration centroids
        concordantLabelCount -> "amout of labels in concordance between the labels stored in pool data and the calculated
                                with new centroid values"
        maxPoolSize          -> "the maximum size of the pool"
    )

Updates the information stored in `centroids`, `labeledData` and `labeledDataLabels` if the concordance isn't 100% with the
new labeled data calculated with the updated centroid values.
"""
function updateInformation(poolData::Array{T, 2} where {T<:Number}, centroids::Array{T, 2} where {T<:Number}, labeledData::Array{T, 2} 
                           where {T<:Number}, labeledDataLabels::Array{T} where {T<:Number}, newLabeledData::Array{T} where {T<:Number}, 
                           currentCentroids::Array{T, 2} where {T<:Number}, intermed::Array{T, 2} where {T<:Number}, 
                           concordantLabelCount::Int64, maxPoolSize::Int64)
    # If there's still some difference (concordantLabelCount/maxPoolSize < 1) means that the classifier elements 
    # (as centroids, data and labels) need to be updated.
    if concordantLabelCount/maxPoolSize < 1
        centroids = vcat(currentCentroids, intermed)

        labeledData = poolData[:, 1:end-1]
        labeledDataLabels = newLabeledData
    end

    return centroids, labeledData, labeledDataLabels
end

end