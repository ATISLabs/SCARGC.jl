using Pkg
Pkg.activate("../")

using StatsPlots, NearestNeighbors, SCARGC

function nearestNeighborStatic(dataset::Array{T, 2} where {T<:Number}, percentTraining::Float64)
    labeledData, labeledDataLabels, streamData, streamLabels, _ = SCARGC.fitData(dataset, percentTraining)
    
    tree = KDTree(permutedims(labeledData))
    accuracyVector = zeros(size(streamData, 1))
    
    for stream = 1:size(streamData, 1)
        index, _ = knn(tree, streamData[stream, :], 1, true)
        
        if (streamLabels[stream] == labeledDataLabels[index[1]])
            global accuracyVector[stream] = 1
        end
    end
    
    finalAccuracy = (sum(accuracyVector)/size(streamData, 1)) * 100
    
    println(finalAccuracy)
    
    return finalAccuracy
end

function nearestNeighborSliding(dataset::Array{T, 2} where {T<:Number}, percentTraining::Float64)
    labeledData, labeledDataLabels, streamData, streamLabels, _ = SCARGC.fitData(dataset, percentTraining)
    
    tree = KDTree(permutedims(labeledData))
    accuracyVector = zeros(size(streamData, 1))
    
    windowSize = 50
    data = zeros(50, size(streamData, 2) + 1)
    dataStreamIndex = 1
    
    for stream = 1:size(streamData, 1)
        index, _ = knn(tree, streamData[stream, :], 1, true)
        
        if (streamLabels[stream] == labeledDataLabels[index[1]])
            global accuracyVector[stream] = 1
        end
        
        data[dataStreamIndex, 1:end-1] = streamData[stream, :]
        data[dataStreamIndex, end] = streamLabels[stream]
        dataStreamIndex += 1
        
        if stream % windowSize == 0
            labeledData = data[:, 1:end-1]
            labeledDataLabels = data[:, end]
            data = zeros(50, size(streamData, 2) + 1)
            dataStreamIndex = 1
            tree = KDTree(permutedims(labeledData))
        end
    end
    
    finalAccuracy = (sum(accuracyVector)/size(streamData, 1)) * 100
    
    println(finalAccuracy)
    
    return finalAccuracy
end

function nearestNeighborLandmark(dataset::Array{T, 2} where {T<:Number}, percentTraining::Float64)
    labeledData, labeledDataLabels, streamData, streamLabels, _ = SCARGC.fitData(dataset, percentTraining)
    
    tree = KDTree(permutedims(labeledData))
    accuracyVector = zeros(size(streamData, 1))
    
    windowSize = 50
    maxWindowSize = 100
    
    data = zeros(100, size(streamData, 2) + 1)
    dataStreamIndex = 1
    
    for stream = 1:size(streamData, 1)
        index, _ = knn(tree, streamData[stream, :], 1, true)
        
        if (streamLabels[stream] == labeledDataLabels[index[1]])
            global accuracyVector[stream] = 1
        end
        
        data[dataStreamIndex, 1:end-1] = streamData[stream, :]
        data[dataStreamIndex, end] = streamLabels[stream]
        dataStreamIndex += 1
        
        if dataStreamIndex == maxWindowSize
            labeledData = data[1:windowSize, 1:end-1]
            labeledDataLabels = data[1:windowSize, end]
            tree = KDTree(permutedims(labeledData))
            
            data[1:windowSize, :] = data[windowSize+1:end, :]
            dataStreamIndex = 50
        end
    end
    
    finalAccuracy = (sum(accuracyVector)/size(streamData, 1)) * 100
    
    println(finalAccuracy)
    
    return finalAccuracy
end

path = "../src/datasets/synthetic/"

datasets = [
    "1CDT.txt", "2CDT.txt", "1CHT.txt", "2CHT.txt", "4CR.txt", "4CRE-V1.txt", 
    "4CRE-V2.txt", "5CVT.txt", "1CSurr.txt", "4CE1CF.txt", "FG_2C_2D.txt", "UG_2C_2D.txt", 
    "UG_2C_3D.txt", "UG_2C_5D.txt", "MG_2C_2D.txt", "GEARS_2C_2D.txt"
]

K_values = [2, 2, 2, 2, 4, 4, 4, 5, 4, 5, 4, 2, 2, 2, 4, 2]

rowCounts = [
    16000, 16000, 16000, 16000, 144400, 125000, 183000, 24000, 55283, 
    173250, 200000, 100000, 200000, 200000, 200000, 200000
]

featureCount = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 2, 2]

poolValues = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]

trainingValues = [
    0.3125, 0.3125, 0.3125, 0.3125, 0.0346, 0.04, 0.0273, 0.125, 
    0.0904, 0.0288, 0.05, 0.025, 0.025, 0.025, 0.025, 0.025
]

x = [
    "1CDT", "2CDT", "1CHT", "2CHT", "4CR", "4CRE-V1", "4CRE-V2", "5CVT", 
    "1CSurr", "4CE1CF", "FG_2C_2D", "UG_2C_2D", "UG_2C_3D", "UG_2C_5D", 
    "MG_2C_2D", "GEARS_2C_2D"
]

accuracies = zeros(16, 4)

for i = 1:4
    data = SCARGC.extractValuesFromFile(path * datasets[i], rowCounts[i], featureCount[i] + 1)
    _, accuracy = SCARGC.scargc_1NN(data, trainingValues[i], poolValues[i], K_values[i])
    ac = nearestNeighborStatic(data, trainingValues[i])
    ac2 = nearestNeighborSliding(data, trainingValues[i])
    ac3 = nearestNeighborLandmark(data, trainingValues[i])
    accuracies[i, 1] = accuracy
    accuracies[i, 2] = ac
    accuracies[i, 3] = ac2
    accuracies[i, 4] = ac3
end

for i = 5:7
    data = SCARGC.extractValuesFromFile(path * datasets[i], rowCounts[i], featureCount[i] + 1)
    _, accuracy = SCARGC.scargc_1NN(data, trainingValues[i], poolValues[i], K_values[i])
    ac = nearestNeighborStatic(data, trainingValues[i])
    ac2 = nearestNeighborSliding(data, trainingValues[i])
    ac3 = nearestNeighborLandmark(data, trainingValues[i])
    accuracies[i, 1] = accuracy
    accuracies[i, 2] = ac
    accuracies[i, 3] = ac2
    accuracies[i, 4] = ac3
end

for i = 8:10
    data = SCARGC.extractValuesFromFile(path * datasets[i], rowCounts[i], featureCount[i] + 1)
    _, accuracy = SCARGC.scargc_1NN(data, trainingValues[i], poolValues[i], K_values[i])
    ac = nearestNeighborStatic(data, trainingValues[i])
    ac2 = nearestNeighborSliding(data, trainingValues[i])
    ac3 = nearestNeighborLandmark(data, trainingValues[i])
    accuracies[i, 1] = accuracy
    accuracies[i, 2] = ac
    accuracies[i, 3] = ac2
    accuracies[i, 4] = ac3
end

for i = 11:13
    data = SCARGC.extractValuesFromFile(path * datasets[i], rowCounts[i], featureCount[i] + 1)
    _, accuracy = SCARGC.scargc_1NN(data, trainingValues[i], poolValues[i], K_values[i])
    ac = nearestNeighborStatic(data, trainingValues[i])
    ac2 = nearestNeighborSliding(data, trainingValues[i])
    ac3 = nearestNeighborLandmark(data, trainingValues[i])
    accuracies[i, 1] = accuracy
    accuracies[i, 2] = ac
    accuracies[i, 3] = ac2
    accuracies[i, 4] = ac3
end

for i = 14:16
    data = SCARGC.extractValuesFromFile(path * datasets[i], rowCounts[i], featureCount[i] + 1)
    _, accuracy = SCARGC.scargc_1NN(data, trainingValues[i], poolValues[i], K_values[i])
    ac = nearestNeighborStatic(data, trainingValues[i])
    ac2 = nearestNeighborSliding(data, trainingValues[i])
    ac3 = nearestNeighborLandmark(data, trainingValues[i])
    accuracies[i, 1] = accuracy
    accuracies[i, 2] = ac
    accuracies[i, 3] = ac2
    accuracies[i, 4] = ac3
end

gr(size=(900,400))
ctg = repeat(["SCARGC_1NN", "Static", "Sliding", "Landmark"], inner=16)
groupedbar(accuracies, group = ctg, xlabel = "Datasets", 
    ylabel = "Accuracy", bar_position = :dodge, w = 0, framestyle = :box, bar_width = 0.67, xticks=(1:16, x), xrotation=30)
