module SCARGC

using Distances

"""
    fitData(
        dataset         -> the whole dataset loaded from file system
        percentTraining -> percentage of the amount of rows to be used as labeled data
    )

Divides the dataset into labeled data and unlabeled data, training and testing data respectively.
These arrays are gonna be used in the whole program, so the fitting part is considered extremelly important because, if something's wrong in the this step, the whole application is gonna fail.

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
    resizeData(
        labeledData -> the labeled data matrix that is going to be resized
        streamData  -> testing data matrix that is going to be resized
        K           -> defined number of clusters
        features    -> total of features, that also is going to be redefined
    )

The function resizes the labeled and stream data with 1s if the feature count is smaller than the value of K. If it's not, the function just returns the original values.
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
If it's value is smaller then the smallest distance, the smallest distance recieves this distance value and the label value, at this index, is stored.
The return of the function is the output and the data from the nearest neighbor.
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

end