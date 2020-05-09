module SCARGC

"""
    fitData(
        dataset      -> the whole dataset loaded from file system
        trainingRows -> percentage of the amount of rows to be used as training
    )

Divides the dataset into labeled data and unlabeled data, training and testing data respectively.
These arrays are gonna be used in the whole program, so the fitting part is considered extremelly important because, if something's wrong in the this step, the whole application is gonna fail.

The return of this function has the following variables:
    - labels         -> all dataset labels
    - training data  -> percentTraining of the dataset features
    - trainingLabels -> also percentTraining of the dataset labels
    - streamData     -> remaining data of the dataset
    - streamLabels   -> remaining labels of the dataset
    - features       -> number of features present in the data
"""
function fitData(dataset::Array{T, 2} where {T<:Number}, percentTraining::Int64)
    rows = size(dataset)[1]
    columns = size(dataset)[2]

    trainingRows = Int((percentTraining/100) * rows)

    trainingData = dataset[1:trainingRows, 1:columns-1]
    trainingLabels = dataset[1:trainingRows, columns]

    streamData = dataset[trainingRows+1:rows, 1:columns-1]
    streamLabels = dataset[trainingRows+1:rows, columns]

    features = columns - 1

    labels = dataset[:, columns]

    return labels, trainingData, trainingLabels, streamData, streamLabels, features
    
end

end