using SCARGC, Test

@testset "Dataset fit" begin
    dataset = rand(10^3, 5)

    labels, trainingData, trainingLabels, streamData, streamLabels, features = SCARGC.fitData(dataset, 5)

    @testset "Arrays size" begin
        @testset "Row count" begin
            @test size(labels)[1] == 10^3
            @test size(trainingData)[1] == 50
            @test size(trainingLabels)[1] == size(trainingData)[1]
            @test size(streamData)[1] == 950
            @test size(streamLabels)[1] == size(streamData)[1]
        end

        @testset "Column count" begin
            @test_throws BoundsError size(labels)[2]
            @test size(trainingData)[2] == 4
            @test_throws BoundsError size(trainingLabels)[2]
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

                @test trainingData[row, column] == dataset[row, column]
                @test trainingLabels[row] == dataset[row, 5]
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