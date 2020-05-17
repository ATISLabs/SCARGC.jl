module SCARGC

using Distances, Statistics
using Clustering

"""
scargc_1NN(
    dataset         -> dataset used in the algorighm
    percentTraining -> amount of data that is goung to be used as training data
    maxPoolSize     -> maximum instances that the pool size can store
    K               -> nuber of clusters
)

SCARGC implementation with the Nearest Neighbor classifier for label predicting.
The function prints the final accuracy and returns the vector with all predicted labels.

The most important function lines are:
    - Separing the dataset into different arrays/elements (line 38)
    - Resizing the data if the feature count is smaller than the K. (line 39)
    - Finding the initial centroids. (line 40)
    - Vector of predicted labels (line 42)
    - Vector to calculate the accuracy at the end of the function. (line 44)
    - The size of the poolData matrix is `maxPoolSize` rows by features on test instance + 1 (the predicted label). (line 45)

    INSIDE THE FOR IF
        - Temporary centroids from current iteration. (line 59)
        - Variables to store the size of the matrixes. (lines 61 and 62)
        - Finding the intermed matrix, with the centroids and labels from current iteration using values from the previous. (line 64)
          one and the labels of those centroids.
        - Centroids from the current iteration. (lines 67, 68 and 69)
        - Size of current centroids' matrix. (line 71)
        - Getting new labeled data with updated centroid values. (line 73)
        - Getting the amount of new labeled data that is the same as the labeled data stored on pool size. (line 75)
        - Updating values (line 77)
        - Reseting poolData (lines 81, 82)
"""
function scargc_1NN(dataset::Array{T, 2} where {T<:Number}, percentTraining::Float64, maxPoolSize::Int64, K::Int64)
    labels, labeledData, labeledDataLabels, streamData, streamLabels, features = fitData(dataset, percentTraining)
    labeledData, streamData, features = resizeData(labeledData, streamData, K, features)
    centroids = findCentroids(labels, features, labeledData, labeledDataLabels, K)

    predictedLabels = zeros(size(streamLabels, 1))

    accuracyVector = zeros(size(streamData, 1))
    poolData = zeros(maxPoolSize, size(streamData, 2) + 1)
    poolDataIndex = 1

    for stream = 1:size(streamData, 1)
        testInstance = streamData[stream, :]
        currentLabel = streamLabels[stream]

        predictedLabel, _ = knnClassification(labeledData, labeledDataLabels, testInstance)

        poolData[poolDataIndex, 1:size(streamData, 2)] = testInstance
        poolData[poolDataIndex, size(streamData, 2) + 1] = predictedLabel
        poolDataIndex += 1

        if (poolDataIndex > maxPoolSize)
            tempCurrentCentroids = kmeans(poolData[:, 1:size(poolData, 2) - 1], K).centers

            sizeTempCurrentCentroids = size(tempCurrentCentroids)
            sizePoolData = size(poolData)

            intermed, centroidLabels = findLabelForCurrentCentroids(centroids, tempCurrentCentroids, K)
            sizeIntermed = size(intermed)

            currentCentroids = zeros(sizeTempCurrentCentroids[1], sizeTempCurrentCentroids[2] + 1)
            currentCentroids[:, 1:sizeTempCurrentCentroids[2]] = tempCurrentCentroids
            currentCentroids[:, size(currentCentroids, 2)] = centroidLabels

            sizeCurrentCentroids = size(currentCentroids)

            newLabeledData = calculateNewLabeledData(poolData, currentCentroids, intermed)
            
            concordantLabelCount = count(label -> label == 1, poolData[:, 1:sizePoolData[2] - 1] .== newLabeledData)

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

    finalAccuracy = (sum(accuracyVector)/size(streamData)[1]) * 100

    println(finalAccuracy)

    return predictedLabels
end

"""
    fitData(
        dataset         -> the whole dataset loaded from file system
        percentTraining -> percentage of the amount of rows to be used as labeled data
    )

Divides the dataset into labeled data and unlabeled data, training and testing data respectively.
These arrays are gonna be used in the whole program, so the fitting part is considered extremelly important because, if 
something's wrong in the this step, the whole application is gonna fail.

The return of this function has the following variables:
    - labels            -> all dataset labels
    - labeledData       -> `percentTraining` of the dataset features
    - labeledDataLabels -> also `percentTraining` of the dataset labels
    - streamData        -> remaining data of the dataset
    - streamLabels      -> remaining labels of the dataset
    - features          -> number of features present in the data
"""
function fitData(dataset::Array{T, 2} where {T<:Number}, percentTraining::Float64)
    rows = size(dataset, 1)
    columns = size(dataset, 2)

    labeledRowCount = Int(ceil((percentTraining/100) * rows))

    labeledData = dataset[1:labeledRowCount, 1:columns - 1]
    labeledDataLabels = dataset[1:labeledRowCount, columns]

    streamData = dataset[labeledRowCount + 1:rows, 1:columns - 1]
    streamLabels = dataset[labeledRowCount + 1:rows, columns]

    features = columns - 1

    labels = dataset[:, columns]

    return labels, labeledData, labeledDataLabels, streamData, streamLabels, features
end

"""
    updateData(
        labeledData -> the labeled data matrix that is going to be resized
        streamData  -> testing data matrix that is going to be resized
        K           -> defined number of clusters
        features    -> total of features, that also is going to be redefined
    )

The function resizes the labeled and stream data with 1s if the feature count is smaller than the value of K. If it's not, the 
function just returns the original values.
After resizing, the function returns the new labeled data, the new stream data and the updated feature count.
"""
function resizeData(labeledData::Array{T, 2} where {T<:Number}, streamData::Array{T, 2} where {T<:Number}, K::Int64, features::Int64)
    if (features < K)
        newLabeledData = ones(size(labeledData, 1), K)
        newStreamData = ones(size(streamData, 1), K)

        newLabeledData[:, 1:size(labeledData,2)] = labeledData
        newStreamData[:, 1:size(streamData, 2)] = streamData

        newFeatureCount = K

        return newLabeledData, newStreamData, newFeatureCount
    else
        return labeledData, streamData, features
    end
end

"""
    knnClassification(
        labeledData  -> labeled data used to apply the euclidean distance
        labels       -> all data labels
        testInstance -> the instance we want to predict the output based on the nearest neighbor.
    )

Applies the Nearest Neighbor to the `testInstance`.
The function calculates, for each `labeledData`, the Euclidean Distance between it and the test instance and. 
If it's value is smaller then the smallest distance, the smallest distance recieves this distance value and the label value, at 
this index, is stored.
The rurn of the function is the output and the data from the nearest neighbor.
"""
function knnClassification(labeledData::Array{T, 2} where {T<:Number}, labels::Array{T} where {T<:Number}, testInstance::Array{T} where {T<:Number})
    output = nothing
    resultData = nothing
    smallerDistance = Inf

    datasetRows = Int(size(labeledData, 1))

    for row = 1:datasetRows
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
        labels            -> whole dataset labels
        features          -> number of features
        labeledData       -> labeled data used in classification
        labeledDataLabels -> labels of the labeled data used in classification
        K                 -> number of clusters
    )

Finds the initial centroids used in program, using KMeans if necessary, with the data already fit. 
The function has two possible ways:
    - The value of K is the same as the number of classes; and
    - The value of K is different as the number of classes.

If the values are the same, the function creates the centroids as the median of the data. In other words, if these numbers are 
equal, the centroids have the value of the median of each feature.
Otherwise, if the values aren't the same, the KMeans method is used to define the initial centroids.

This function returns the centroids array.
"""
function findCentroids(labels::Array{T} where {T<:Number}, features::Int64, labeledData::Array{T, 2} where {T<:Number}, labeledDataLabels::Array{T} where {T<:Number}, K::Int64)
    classes = unique(labels)
    totalClasses = size(classes, 1)

    centroids = nothing

    # Finding the first centroids
    # If the number of classes is the same as K, the initial centroid of each class is the mean of each feature.
    # If it's not, a clustering is necessary to find the centroids.
    if K == totalClasses
        println("(K == C)")
        global centroids = zeros(totalClasses, features + 1)

        for class = 1:totalClasses
            tempCentroids = zeros(features)

            for feature = 1:features
                tempCentroids[feature] = median(labeledData[findall(label -> label == classes[class], labeledDataLabels), feature])
            end
            
            centroids[class, 1:features] = tempCentroids
        end

        global centroids[:, features + 1] = classes
    else
        println("(K != C)")
        tempCentroids = kmeans(labeledData, K).centers

        centroidLabels = zeros(size(tempCentroids, 1))

        for row = 1:size(tempCentroids, 1)
            output, _ = knnClassification(labeledData, labeledDataLabels, tempCentroids[row, :])
            global centroidLabels[row] = output
        end

        global centroids = zeros(size(tempCentroids, 1), size(tempCentroids, 2) + 1)
        global centroids[:, 1:size(tempCentroids, 2)] = tempCentroids
        global centroids[:, size(centroids, 2)] = centroidLabels
    end

    return centroids
end

"""
    findLabelForCurrentCentroids(
        pastCentroids    -> centroids from previous iteration
        currentCentroids -> centroids from current iteration
        K                -> number of clusters
    )

The function uses the last iteration centroids' labels to define the current iteration centroids' labels.
Given the current centroids from the most recent unlabeled clusters and the past centroids from the previously labeled clusters, 
each centroid from the past iteration have a label yi and each centroid from current iteration needs a label ˆyi. This label is 
obtained by aplying nearest neighbor algorithm. In other words, the label given to a centroid in the current iteration is the 
same as the label given to it's nearest neighbor in the past iteration.

The function returns two arrays:
    intermed       -> matrix storing the median between the nearest neighbor's data and the current centroids and it's labels
    centroidLabels -> labels got for the current centroids
"""
function findLabelForCurrentCentroids(pastCentroids::Array{T, 2} where {T<:Number}, currentCentroids::Array{T, 2} where {T<:Number}, K::Int64)
    pastCentroidsSize = size(pastCentroids)
    currentCentroidsSize = size(currentCentroids)

    intermed = zeros(currentCentroidsSize[1], K + 1)
    centroidLabels = zeros(currentCentroidsSize[1])

    for row = 1:currentCentroidsSize[1]
        label, nearestData = knnClassification(pastCentroids[:, 1:pastCentroidsSize[2] - 1], pastCentroids[:, pastCentroidsSize[2]], currentCentroids[row, :])
        
        medianMatrix = zeros(2, currentCentroidsSize[2])
        medianMatrix[1, :] = nearestData
        medianMatrix[2, :] = currentCentroids[row, :]

        global intermed[row, 1:K] = median(medianMatrix, dims=1)
        global intermed[row, K + 1] = label

        global centroidLabels[row] = label
    end

    return intermed, centroidLabels
end

"""
    calculateNewLabeledData(
        poolData          -> data stored in the pool
        currentCentroids  -> centroids from current iteration
        intermedCentroids -> matrix with centroids and their labels 
    )

Given the updated centroid values and the pool data, the function calculates the new labeled data using nearest neighbor over the
vcat of current centroids and the intermed centroids and the pool data. 
The result is going to have the labels got with new centroid values, from the updated classifier.
"""
function calculateNewLabeledData(poolData::Array{T, 2} where {T<:Number}, currentCentroids::Array{T, 2} where {T<:Number}, intermedCentroids::Array{T, 2} where {T<:Number})
    poolDataSize = size(poolData)
    currentCentroidsSize = size(currentCentroids)
    intermedSize = size(intermedCentroids)

    newLabeledData = zeros(poolDataSize[1])

    for row = 1:poolDataSize[1]
        output, _ = knnClassification(vcat(currentCentroids[:, 1:currentCentroidsSize[2] - 1], intermedCentroids[:, 1:intermedSize[2] - 1]), 
        vcat(currentCentroids[:, currentCentroidsSize[2]], intermedCentroids[:, intermedSize[2]]), 
        poolData[row, 1:poolDataSize[2] - 1])

        global newLabeledData[row] = output
    end

    return newLabeledData
end

"""
    updateInformation(
        poolData             -> test instances with it's labels 
        centroids            -> centroids from past iteration
        labeledData          -> labeled data used in classification
        labeledDataLabels    -> labels from the data in `labeledData`
        newLabeledData       -> data calculated with the new centroid values
        currentCentroids     -> centroids from the current iteration
        intermed             -> matrix with all median data between past and current iteration centroids
        concordantLabelCount -> amout of labels in concordance between the labels stored in pool data and the calculated
                                with new centroid values
        maxPoolSize          -> the maximum size of the pool
    )

Updates the information stored in `centroids`, `labeledData` and `labeledDataLabels` if the concordance isn't 100% with the
new labeled data calculated with the updated centroid values.
"""
function updateInformation(poolData::Array{T, 2} where {T<:Number}, centroids::Array{T, 2} where {T<:Number}, labeledData::Array{T, 2} 
    where {T<:Number}, labeledDataLabels::Array{T} where {T<:Number}, newLabeledData::Array{T} where {T<:Number}, currentCentroids::Array{T, 2} where {T<:Number}, 
    intermed::Array{T, 2} where {T<:Number}, concordantLabelCount::Int64, maxPoolSize::Int64)

    # Variables to store the size of the matrixes.
    sizePoolData = size(poolData)

    # If there's still some difference (concordantLabelCount/maxPoolSize < 1) or if the amount of labeled data labels is smaller than the pool data 
    # means that the classifier elements (as centroids, data and labels) need to be updated.
    if concordantLabelCount/maxPoolSize < 1 || size(labeledDataLabels, 1) < sizePoolData[1]
        poolData[:, sizePoolData[2]] = newLabeledData
        
        centroids = nothing
        centroids = vcat(currentCentroids, intermed)

        labeledData = nothing
        labeledData = poolData[:, 1:sizePoolData[2] - 1]

        labeledDataLabels = nothing
        labeledDataLabels = poolData[:, sizePoolData[2]]
    end

    return centroids, labeledData, labeledDataLabels
end

end