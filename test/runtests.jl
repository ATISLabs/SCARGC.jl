using SCARGC, Test, PyCall

@testset "Dataset fit" begin
    dataset = rand(10^3, 5)

    labeledData, labeledDataLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 5.0)

    @testset "Arrays size" begin
        @testset "Row count" begin
            @test size(labeledData)[1] == 50
            @test size(labeledDataLabels)[1] == size(labeledData)[1]
            @test size(streamData)[1] == 950
            @test size(streamLabels)[1] == size(streamData)[1]
        end

        @testset "Column count" begin
            @test size(labeledData)[2] == 4
            @test_throws BoundsError size(labeledDataLabels)[2]
            @test size(streamData)[2] == 4
            @test_throws BoundsError size(streamLabels)[2]
        end
    end

    @testset "Array content" begin
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

    labeledData, labeledDataLabels, streamData, _, _ = SCARGC.fitData(dataset, 20.0)

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
        trainingData, trainingLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 100.0)
        
        centroids = SCARGC.findCentroids(features, trainingData, trainingLabels, 4)

        @testset "Array size" begin
            @test size(centroids) == (4, 3)
        end

        @testset "Array content" begin
            @test centroids[1, :] == [-1.0525, -9.2088, 2.0]
            @test centroids[2, :] == [10.0244, -0.303346, 4.0]
            @test centroids[3, :] == [-9.5809, -0.46916, 3.0]
            @test centroids[4, :] == [-0.0534785, 9.92035, 1.0]
        end
    end

    @testset "K != C" begin
        py"""
            from sklearn.cluster import KMeans
            import numpy as np
            import sys, os

            def kmeans(X, k, init, centroids=[]):
                if (init == 0):
                    return KMeans(n_clusters=k).fit(X).cluster_centers_
                else:
                    centroids = np.array(centroids)
                    sys.stdout = open(os.devnull, 'w')
                    return KMeans(n_clusters=k, init=centroids).fit(X).cluster_centers_"""

        trainingData, trainingLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 100.0)
        
        centroids = SCARGC.findCentroids(features, trainingData, trainingLabels, 2)

        @testset "Array size" begin
            @test size(centroids) == (2, 3)
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

    _, centroidLabels = SCARGC.findLabelForCurrentCentroids(pastCentroids, currentCentroids)

    for row = 1:20
        @test centroidLabels[row] == labels[row]
    end
end

@testset "Claculating new labeled data" begin
    poolData = [
        1.1401 -9.9543 2;
        11.135 -0.622 4;
        9.5399 0.0086581 4;
        -8.5245 -0.98091 3;
        10.476 5.2178 4;
        -2.1146 -10.042 2;
        -5.1069 -2.0286 3;
        10.077 -1.1442 4;
        -1.0195 -8.5242 2;
        -2.015 10.026 1
    ]

    currentCentroids = [
        2.8958 -7.4953 2; 8.2872 -0.68142 4; 6.3705 -1.0572 4; -10.777 1.6717 3;
        -3.9285 9.8147 1; -8.7798 -0.88944 3; -13.005 -0.092225 3; -1.8922 -7.7006 2;
        -1.5711 -7.7011 2; -1.3478 -7.427 2; -3.0136 13.741 1; -7.443 1.0671 3;
        -1.222 -10.88 2; 0.97767 9.6513 1; 0.42688 -12.291 2; 0.54482 -9.2378 2;
        10.375 -1.8902 4; -9.5809 0.40066 3; 10.736 1.6078 4; 14.971 2.2448 4
    ]

    intermedCentroids = [
        0.071359 -6.9043 2; -11.347 -1.6508 3; 3.2586 8.8188 1; 8.5209 -1.297 4;
        -11.167 -1.3958 3; -9.0726 1.4411 3; 0.25484 10.488 1; -1.0525 -9.2088 2;
        -0.22116 11.641 1; -0.048457 14.003 1; -10.575 1.8298 3; -0.0585 9.7192 1;
        10.158 -0.55542 4; 0.18513 8.3579 1; -13.068 -0.46916 3; 9.2202 2.4973 4;
        9.2265 0.12338 4; 9.9718 -0.051272 4; -8.4675 -0.85861 3; -1.3546 -12.383 2
    ]

    expectedLabels = [2, 4, 4, 3, 4, 2, 3, 4, 2, 1]

    testLabeledData = SCARGC.calculateNewLabeledData(poolData, currentCentroids, intermedCentroids)

    @testset "Array size" begin
        @test size(testLabeledData) == size(expectedLabels)
    end

    @testset "Array content" begin
        @test testLabeledData == expectedLabels
    end
end

@testset "Information update" begin
    poolData = [
        -8.7798 -0.88944 1.0 1.0 3.0; -13.005 -0.092225 1.0 1.0 3.0; -1.8922 -7.7006 1.0 1.0 2.0; -1.5711 -7.7011 1.0 1.0 2.0; -1.3478 -7.427 1.0 1.0 2.0; 
        -3.0136 13.741 1.0 1.0 1.0; -7.443 1.0671 1.0 1.0 3.0; -1.222 -10.88 1.0 1.0 2.0; 0.97767 9.6513 1.0 1.0 1.0; 0.42688 -12.291 1.0 1.0 2.0; 
        0.54482 -9.2378 1.0 1.0 2.0; 10.375 -1.8902 1.0 1.0 4.0; -9.5809 0.40066 1.0 1.0 3.0; 10.736 1.6078 1.0 1.0 4.0; 14.971 2.2448 1.0 1.0 4.0; 
        0.071359 -6.9043 1.0 1.0 2.0; -11.347 -1.6508 1.0 1.0 3.0; 3.2586 8.8188 1.0 1.0 1.0; 8.5209 -1.297 1.0 1.0 4.0; -11.167 -1.3958 1.0 1.0 3.0
    ]

    centroids = [2.8958 -7.4953 1.0 1.0 2.0; 7.32885 -0.86931 1.0 1.0 4.0; -10.777 1.6717 1.0 1.0 3.0; -3.9285 9.8147 1.0 1.0 1.0]

    labeledData = [2.8958 -7.4953 1.0 1.0; 8.2872 -0.68142 1.0 1.0; 6.3705 -1.0572 1.0 1.0; -10.777 1.6717 1.0 1.0; -3.9285 9.8147 1.0 1.0]

    labeledDataLabels = [2.0, 4.0, 4.0, 3.0, 1.0]

    newLabeledData = [3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0, 4.0, 3.0]

    currentCentroids = [
        -8.7798 -0.88944 1.0 1.0 3.0; -13.005 -0.092225 1.0 1.0 3.0; -1.8922 -7.7006 1.0 1.0 2.0; -1.5711 -7.7011 1.0 1.0 2.0; -1.3478 -7.427 1.0 1.0 2.0; 
        -3.0136 13.741 1.0 1.0 1.0; -7.443 1.0671 1.0 1.0 3.0; -1.222 -10.88 1.0 1.0 2.0; 0.97767 9.6513 1.0 1.0 1.0; 0.42688 -12.291 1.0 1.0 2.0; 
        0.54482 -9.2378 1.0 1.0 2.0; 10.375 -1.8902 1.0 1.0 4.0; -9.5809 0.40066 1.0 1.0 3.0; 10.736 1.6078 1.0 1.0 4.0; 14.971 2.2448 1.0 1.0 4.0; 
        0.071359 -6.9043 1.0 1.0 2.0; -11.347 -1.6508 1.0 1.0 3.0; 3.2586 8.8188 1.0 1.0 1.0; 8.5209 -1.297 1.0 1.0 4.0; -11.167 -1.3958 1.0 1.0 3.0
    ]

    intermed = [
        -9.7784 0.39113 1.0 1.0 3.0; -11.891 0.7897375 1.0 1.0 3.0; 0.5017999999999999 -7.59795 1.0 1.0 2.0; 0.66235 -7.5982 1.0 1.0 2.0; 0.7739999999999999 -7.46115 1.0 1.0 2.0; 
        -3.47105 11.77785 1.0 1.0 1.0; -9.11 1.3694 1.0 1.0 3.0; 0.8369 -9.187650000000001 1.0 1.0 2.0; -1.475415 9.733 1.0 1.0 1.0; 1.66134 -9.89315 1.0 1.0 2.0; 
        1.72031 -8.36655 1.0 1.0 2.0; 8.851925 -1.379755 1.0 1.0 4.0; -10.17895 1.0361799999999999 1.0 1.0 3.0; 9.032425 0.36924499999999993 1.0 1.0 4.0; 11.149925 0.687745 1.0 1.0 4.0; 
        1.4835795 -7.1998 1.0 1.0 2.0; -11.062 0.01044999999999996 1.0 1.0 3.0; -0.3349500000000001 9.316749999999999 1.0 1.0 1.0; 7.924875 -1.083155 1.0 1.0 4.0; -10.972 0.13795000000000002 1.0 1.0 3.0
    ]

    maxPoolSize = 20

    @testset "concordantLabelCount/maxPoolSize >= 1" begin
        concordantLabelCount = 20
        labeledDataLabels = [2.0, 4.0, 4.0, 3.0, 1.0, 2.0, 4.0, 4.0, 3.0, 1.0, 2.0, 4.0, 4.0, 3.0, 1.0, 2.0, 4.0, 4.0, 3.0, 1.0, 1.0]

        centroidsGot, labeledDataGot, labeledDataLabelsGot = SCARGC.updateInformation(poolData, centroids, labeledData, labeledDataLabels, newLabeledData, 
                                                                                    currentCentroids, intermed, concordantLabelCount, maxPoolSize)

        @testset "Array size" begin
            @test size(centroidsGot) == size(centroids)
            @test size(labeledDataGot) == size(labeledData)
            @test size(labeledDataLabelsGot) == size(labeledDataLabels)
        end

        @testset "Array content" begin
            @test centroidsGot == centroids
            @test labeledDataGot == labeledData
            @test labeledDataLabelsGot == labeledDataLabels
        end
    end

    @testset "concordantLabelCount/maxPoolSize < 1" begin
        labeledDataLabels = [2.0, 4.0, 4.0, 3.0, 1.0]
        concordantLabelCount = 6

        expectedCentroids = [
            -8.7798 -0.88944 1.0 1.0 3.0; -13.005 -0.092225 1.0 1.0 3.0; -1.8922 -7.7006 1.0 1.0 2.0; -1.5711 -7.7011 1.0 1.0 2.0; -1.3478 -7.427 1.0 1.0 2.0; 
            -3.0136 13.741 1.0 1.0 1.0; -7.443 1.0671 1.0 1.0 3.0; -1.222 -10.88 1.0 1.0 2.0; 0.97767 9.6513 1.0 1.0 1.0; 0.42688 -12.291 1.0 1.0 2.0; 
            0.54482 -9.2378 1.0 1.0 2.0; 10.375 -1.8902 1.0 1.0 4.0; -9.5809 0.40066 1.0 1.0 3.0; 10.736 1.6078 1.0 1.0 4.0; 14.971 2.2448 1.0 1.0 4.0; 
            0.071359 -6.9043 1.0 1.0 2.0; -11.347 -1.6508 1.0 1.0 3.0; 3.2586 8.8188 1.0 1.0 1.0; 8.5209 -1.297 1.0 1.0 4.0; -11.167 -1.3958 1.0 1.0 3.0;
            -9.7784 0.39113 1.0 1.0 3.0; -11.891 0.7897375 1.0 1.0 3.0; 0.5017999999999999 -7.59795 1.0 1.0 2.0; 0.66235 -7.5982 1.0 1.0 2.0; 0.7739999999999999 -7.46115 1.0 1.0 2.0; 
            -3.47105 11.77785 1.0 1.0 1.0; -9.11 1.3694 1.0 1.0 3.0; 0.8369 -9.187650000000001 1.0 1.0 2.0; -1.475415 9.733 1.0 1.0 1.0; 1.66134 -9.89315 1.0 1.0 2.0; 
            1.72031 -8.36655 1.0 1.0 2.0; 8.851925 -1.379755 1.0 1.0 4.0; -10.17895 1.0361799999999999 1.0 1.0 3.0; 9.032425 0.36924499999999993 1.0 1.0 4.0; 11.149925 0.687745 1.0 1.0 4.0; 
            1.4835795 -7.1998 1.0 1.0 2.0; -11.062 0.01044999999999996 1.0 1.0 3.0; -0.3349500000000001 9.316749999999999 1.0 1.0 1.0; 7.924875 -1.083155 1.0 1.0 4.0; -10.972 0.13795000000000002 1.0 1.0 3.0
        ]

        expectedLabeledData = [
            -8.7798 -0.88944 1.0 1.0; -13.005 -0.092225 1.0 1.0; -1.8922 -7.7006 1.0 1.0; -1.5711 -7.7011 1.0 1.0; -1.3478 -7.427 1.0 1.0; 
            -3.0136 13.741 1.0 1.0; -7.443 1.0671 1.0 1.0; -1.222 -10.88 1.0 1.0; 0.97767 9.6513 1.0 1.0; 0.42688 -12.291 1.0 1.0; 
            0.54482 -9.2378 1.0 1.0; 10.375 -1.8902 1.0 1.0; -9.5809 0.40066 1.0 1.0; 10.736 1.6078 1.0 1.0; 14.971 2.2448 1.0 1.0; 
            0.071359 -6.9043 1.0 1.0; -11.347 -1.6508 1.0 1.0; 3.2586 8.8188 1.0 1.0; 8.5209 -1.297 1.0 1.0; -11.167 -1.3958 1.0 1.0
        ]

        expectedLabeledDataLabels = [3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0, 4.0, 3.0]

        centroidsGot, labeledDataGot, labeledDataLabelsGot = SCARGC.updateInformation(poolData, centroids, labeledData, labeledDataLabels, newLabeledData, 
                                                                                    currentCentroids, intermed, concordantLabelCount, maxPoolSize)

        @testset "Array size" begin
            @test size(centroidsGot, 1) == size(currentCentroids, 1) + size(intermed, 1)
            @test size(labeledDataGot, 1) == size(poolData, 1)
            @test size(labeledDataLabelsGot, 1) == size(poolData, 1)

            @test size(centroidsGot, 2) == size(currentCentroids, 2) == size(intermed, 2)
            @test size(labeledDataGot, 2) == size(poolData, 2) - 1
        end

        @testset "Array content" begin
            @test centroidsGot == expectedCentroids
            @test labeledDataGot == expectedLabeledData
            @test labeledDataLabelsGot == expectedLabeledDataLabels
        end
    end
end

@testset "SCARGC with 1NN classifier" begin
    dataset = [
        2.8958 -7.4953 2; 8.2872 -0.68142 4; 6.3705 -1.0572 4;-10.777 1.6717 3;-3.9285 9.8147 1;-8.7798 -0.88944 3;-13.005 -0.092225 3;-1.8922 -7.7006 2;-1.5711 -7.7011 2;-1.3478 -7.427 2;
        -3.0136 13.741 1;-7.443 1.0671 3;-1.222 -10.88 2;0.97767 9.6513 1;0.42688 -12.291 2;0.54482 -9.2378 2;10.375 -1.8902 4;-9.5809 0.40066 3;10.736 1.6078 4;14.971 2.2448 4;
        0.071359 -6.9043 2;-11.347 -1.6508 3;3.2586 8.8188 1;8.5209 -1.297 4;-11.167 -1.3958 3;-9.0726 1.4411 3;0.25484 10.488 1;-1.0525 -9.2088 2;-0.22116 11.641 1;-0.048457 14.003 1;
        -10.575 1.8298 3;-0.0585 9.7192 1;10.158 -0.55542 4;0.18513 8.3579 1;-13.068 -0.46916 3;9.2202 2.4973 4;9.2265 0.12338 4;9.9718 -0.051272 4;-8.4675 -0.85861 3;-1.3546 -12.383 2;
        1.1401 -9.9543 2;11.135 -0.622 4;9.5399 0.0086581 4;-8.5245 -0.98091 3;10.476 5.2178 4;-2.1146 -10.042 2;-5.1069 -2.0286 3;10.077 -1.1442 4;-1.0195 -8.5242 2;-2.015 10.026 1;
        2.2047 12.298 1;-0.074896 10.403 1;0.90863 -7.3251 2;2.5506 10.718 1;-1.7999 -11.768 2;10.041 1.9354 4;0.71283 -10.392 2;-0.24533 12.384 1;-10.916 -3.0079 3;13.157 -0.087517 4;
        12.36 -2.1673 4;12.158 1.6823 4;8.3389 1.1933 4;0.51037 -11.019 2;0.39083 6.5315 1;-1.277 -12.846 2;9.0778 0.36851 4;-12.049 2.7196 3;-12.677 -2.1093 3;-10.661 0.84432 3;
        -0.36669 10.772 1;1.8032 6.9087 1;-1.3556 -8.8028 2;-0.84456 -7.1571 2;-0.55158 8.9382 1;-1.9958 -7.0433 2;-1.7224 -15.051 2;-1.406 -10.686 2;2.5305 11.46 1;1.0389 -9.7381 2;
        -11.587 1.5958 3;0.28096 8.516 1;-0.00018349 9.131 1;-10.635 2.8625 3;-2.0658 11.189 1;0.98778 10.147 1;-10.577 -0.26176 3;-1.5552 -9.5398 2;1.995 9.5663 1;-8.5554 3.3411 3;
        -1.4494 10.32 1;2.2654 12.618 1;10.908 -0.56058 4;2.5514 10.125 1;9.4298 0.82499 4;7.7632 0.24353 4;10.986 0.72106 4;-5.187 1.0313 3;-10.733 2.1273 3;8.7337 1.0812 4
    ]

    expectedLabels = [
        1, 3, 2, 1, 2, 2, 4, 3, 4, 4, 2, 3, 1, 4, 3, 3, 1, 2, 1, 1, 3, 1, 4, 1, 3, 4, 4, 4, 3, 2, 
        2, 4, 4, 3, 4, 2, 3, 4, 2, 1, 1, 1, 2, 1, 2, 4, 2, 1, 3, 4, 4, 4, 4, 2, 1, 2, 4, 3, 3, 3, 
        1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 3, 1, 1, 3, 1, 1, 3, 2, 1, 3, 1, 1, 4, 1, 4, 4, 4, 3, 3, 4
    ]

    labelsGot = SCARGC.scargc_1NN(dataset, 10.0, 20, 4)

    @test (count(label -> label == 1, labelsGot .== expectedLabels)/90) * 100 >= 90
end