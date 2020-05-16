using SCARGC, Test, Clustering

@testset "Dataset fit" begin
    dataset = rand(10^3, 5)

    labels, labeledData, labeledDataLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 5.0)

    @testset "Arrays size" begin
        @testset "Row count" begin
            @test size(labels)[1] == 10^3
            @test size(labeledData)[1] == 50
            @test size(labeledDataLabels)[1] == size(labeledData)[1]
            @test size(streamData)[1] == 950
            @test size(streamLabels)[1] == size(streamData)[1]
        end

        @testset "Column count" begin
            @test_throws BoundsError size(labels)[2]
            @test size(labeledData)[2] == 4
            @test_throws BoundsError size(labeledDataLabels)[2]
            @test size(streamData)[2] == 4
            @test_throws BoundsError size(streamLabels)[2]
        end
    end

    @testset "Array content" begin
        @testset "Labels" begin
            for cases = 1:100
                row = rand(1:1000)

                @test labels[row] == dataset[row, 5]
            end
        end

        @testset "Training data" begin
            for cases = 1:100
                row = rand(1:50)
                column = rand(1:4)

                @test labeledData[row, column] == dataset[row, column]
                @test labeledDataLabels[row] == dataset[row, 5]
            end
        end

        @testset "Stream data" begin
            for cases = 1:100
                streamRow = rand(1:950)
                row = streamRow + 50
                column = rand(1:4)

                @test streamData[streamRow, column] == dataset[row, column]
                @test streamLabels[streamRow] == dataset[row, 5]
            end
        end
    end

    @testset "Feature count" begin
        @test features == 4
    end
end

@testset "Data resizing" begin
    @testset "features >= K" begin
        dataset = rand(10^3, 5)
        labels, labeledData, labeledDataLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 5.0)
        labeledData, streamData, features = SCARGC.resizeData(labeledData, streamData, 4, features)

        @test size(labeledData, 2) == 4
        @test size(streamData, 2) == 4
        @test features == 4
    end

    @testset "features < K" begin
        dataset = rand(10^3, 3)
        labels, labeledData, labeledDataLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 5.0)
        labeledData, streamData, features = SCARGC.resizeData(labeledData, streamData, 4, features)

        @testset "Arrays size" begin
            @test size(labeledData, 2) == 4
            @test size(streamData, 2) == 4
            @test features == 4
        end

        @testset "Updated values" begin
            @test labeledData[:, 3:4] == ones(size(labeledData, 1), 2)
            @test streamData[:, 3:4] == ones(size(streamData, 1), 2)
            @test features == 4
        end
    end
end

@testset "KNN Classification" begin
    dataset = [
        2.8958 -7.4953 2; 8.2872 -0.68142 4; 6.3705 -1.0572 4; -10.777 1.6717 3; -3.9285 9.8147 1;
        -8.7798 -0.88944 3; -13.005 -0.092225 3; -1.8922 -7.7006 2; -1.5711 -7.7011 2; -1.3478 -7.427 2;
        -3.0136 13.741 1; -7.443 1.0671 3; -1.222 -10.88 2; 0.97767 9.6513 1; 0.42688 -12.291 2;
        0.54482 -9.2378 2; 10.375 -1.8902 4; -9.5809 0.40066 3; 10.736 1.6078 4; 14.971 2.2448 4;
        0.071359 -6.9043 2; -11.347 -1.6508 3; 3.2586 8.8188 1; 8.5209 -1.297 4; -11.167 -1.3958 3;
        -9.0726 1.4411 3; 0.25484 10.488 1; -1.0525 -9.2088 2; -0.22116 11.641 1; -0.048457 14.003 1;
        -10.575 1.8298 3; -0.0585 9.7192 1; 10.158 -0.55542 4; 0.18513 8.3579 1; -13.068 -0.46916 3;
        9.2202 2.4973 4; 9.2265 0.12338 4; 9.9718 -0.051272 4; -8.4675 -0.85861 3; -1.3546 -12.383 2;  
        1.1401 -9.9543 2; 11.135 -0.622 4; 9.5399 0.0086581 4; -8.5245 -0.98091 3; 10.476 5.2178 4;
        -2.1146 -10.042 2; -5.1069 -2.0286 3; 10.077 -1.1442 4; -1.0195 -8.5242 2; -2.015 10.026 1
    ]

    nearestNeighbors = [5, 6, 9, 5, 9, 9, 2, 6, 2, 2, 10, 7, 5, 2, 7, 4, 5, 9, 5, 5, 4, 5, 2, 5, 7, 2, 2, 2, 6, 9, 1, 2, 2, 6, 2, 8, 6, 2, 9, 5]

    _, labeledData, labeledDataLabels, streamData, _, _ = SCARGC.fitData(dataset, 20.0)

    @testset "Output concordance" begin
        for row = 1:40
            output, _ = SCARGC.knnClassification(labeledData, labeledDataLabels, streamData[row, :])

            @test output == dataset[nearestNeighbors[row], 3]
        end        
    end

    @testset "Data equivalence" begin
        for row = 1:40
            _, data = SCARGC.knnClassification(labeledData, labeledDataLabels, streamData[row, :])

            @test data == dataset[nearestNeighbors[row], 1:2]
        end        
    end

end

@testset "Finding centroids" begin
    dataset = [
        2.8958 -7.4953 2; 8.2872 -0.68142 4; 6.3705 -1.0572 4; -10.777 1.6717 3; -3.9285 9.8147 1;
        -8.7798 -0.88944 3; -13.005 -0.092225 3; -1.8922 -7.7006 2; -1.5711 -7.7011 2; -1.3478 -7.427 2;
        -3.0136 13.741 1; -7.443 1.0671 3; -1.222 -10.88 2; 0.97767 9.6513 1; 0.42688 -12.291 2;
        0.54482 -9.2378 2; 10.375 -1.8902 4; -9.5809 0.40066 3; 10.736 1.6078 4; 14.971 2.2448 4;
        0.071359 -6.9043 2; -11.347 -1.6508 3; 3.2586 8.8188 1; 8.5209 -1.297 4; -11.167 -1.3958 3;
        -9.0726 1.4411 3; 0.25484 10.488 1; -1.0525 -9.2088 2; -0.22116 11.641 1; -0.048457 14.003 1;
        -10.575 1.8298 3; -0.0585 9.7192 1; 10.158 -0.55542 4; 0.18513 8.3579 1; -13.068 -0.46916 3;
        9.2202 2.4973 4; 9.2265 0.12338 4; 9.9718 -0.051272 4; -8.4675 -0.85861 3; -1.3546 -12.383 2;  
        1.1401 -9.9543 2; 11.135 -0.622 4; 9.5399 0.0086581 4; -8.5245 -0.98091 3; 10.476 5.2178 4;
        -2.1146 -10.042 2; -5.1069 -2.0286 3; 10.077 -1.1442 4; -1.0195 -8.5242 2; -2.015 10.026 1
    ]

    @testset "K == C" begin
        labels, trainingData, trainingLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 100.0)
        
        trainingData, streamData, features = SCARGC.resizeData(trainingData, streamData, 4, features)
        centroids = SCARGC.findCentroids(labels, features, trainingData, trainingLabels, 4)

        @testset "Array size" begin
            @test size(centroids) == (4, 5)
        end

        @testset "Array content" begin
            @test centroids[1, :] == [-1.0525, -9.2088, 1.0, 1.0, 2.0]
            @test centroids[2, :] == [10.0244, -0.303346, 1.0, 1.0, 4.0]
            @test centroids[3, :] == [-9.5809, -0.46916, 1.0, 1.0, 3.0]
            @test centroids[4, :] == [-0.0534785, 9.92035, 1.0, 1.0, 1.0]
        end
    end

    @testset "K != C" begin
        labels, trainingData, trainingLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 100.0)
            
        trainingData, streamData, features = SCARGC.resizeData(trainingData, streamData, 2, features)
        centroids = SCARGC.findCentroids(labels, features, trainingData, trainingLabels, 2)
        
        testCentroids = kmeans(trainingData, 2).centers
        testOutputs = zeros(50)

        for testRow = 1:50
            testOutputs[testRow], _ = SCARGC.knnClassification(trainingData, trainingLabels, testCentroids[testRow, :])
        end

        @testset "Array size" begin
            @test size(centroids) == (50, 3)
        end

        @testset "Array content" begin
            for row = 1:50
                @test centroids[row, 1:2] == testCentroids[row, :]
                @test centroids[row, 3] == testOutputs[row]
            end
        end
    end
end

@testset "Finding labels for current centroids" begin
    pastCentroids = [-1.45945 -7.59795 1.0 1.0 2.0; 7.32885 -0.86931 1.0 1.0 4.0; -10.777 -0.092225 1.0 1.0 3.0; -3.9285 9.8147 1.0 1.0 1.0]
    currentCentroids = [
        -3.0136 13.741 1.0 1.0; -7.443 1.0671 1.0 1.0; -1.222 -10.88 1.0 1.0; 0.97767 9.6513 1.0 1.0; 0.42688 -12.291 1.0 1.0; 
        0.54482 -9.2378 1.0 1.0; 10.375 -1.8902 1.0 1.0; -9.5809 0.40066 1.0 1.0; 10.736 1.6078 1.0 1.0; 14.971 2.2448 1.0 1.0; 
        0.071359 -6.9043 1.0 1.0; -11.347 -1.6508 1.0 1.0; 3.2586 8.8188 1.0 1.0; 8.5209 -1.297 1.0 1.0; -11.167 -1.3958 1.0 1.0; 
        -9.0726 1.4411 1.0 1.0; 0.25484 10.488 1.0 1.0; -1.0525 -9.2088 1.0 1.0; -0.22116 11.641 1.0 1.0; -0.048457 14.003 1.0 1.0
    ]

    labels = [1.0, 3.0, 2.0, 1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0, 4.0, 3.0, 3.0, 1.0, 2.0, 1.0, 1.0]

    _, centroidLabels = SCARGC.findLabelForCurrentCentroids(pastCentroids, currentCentroids, 4)

    for row = 1:20
        @test centroidLabels[row] == labels[row]
    end
end