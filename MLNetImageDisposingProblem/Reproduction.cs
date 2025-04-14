using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Xunit;

namespace MLNetImageDisposingProblem;

public class InputData
{
    [ImageType(1, 1)] // size doesn't matter here
    public MLImage SourceImage { get; set; }

    public string LabelValue { get; set; }
}

public class OutputData
{
    public string PredictedLabelValue { get; set; }
    public float[] Score { get; set; }
}

public class Reproduction
{
    [Fact]
    public void Throws()
    {
        // throws when transformerShouldResizeToWidthAndHeight == createRandomImagesWithWidthAndHeight
        const int transformerShouldResizeToWidthAndHeight = 2;
        const int createRandomImagesWithWidthAndHeight = 2;
        
        var mlContext = new MLContext();
        var pipeline = CreatePipeline(mlContext, transformerShouldResizeToWidthAndHeight);

        var (trainingInputDatas, trainingInputDatasDataView) = CreateRandomInputData(2, mlContext, createRandomImagesWithWidthAndHeight);
        var transformer = pipeline.Fit(trainingInputDatasDataView);

        var (testInputDatas, testInputDatasDataView) = CreateRandomInputData(2, mlContext, createRandomImagesWithWidthAndHeight);
        var testOutputDataView = transformer.Transform(testInputDatasDataView);

        var h0 = testInputDatas[0].SourceImage.Height;
        
        // I just want to look at the output or output to the log
        var outputDatas = mlContext.Data.CreateEnumerable<OutputData>(testOutputDataView, true).ToArray();

        var ex1 = Assert.Throws<InvalidOperationException>(() =>
        {
            var h1 = testInputDatas[0].SourceImage.Height;
        });
        Assert.Equal("Object is disposed.", ex1.Message); //at Microsoft.ML.Data.MLImage.ThrowInvalidOperationExceptionIfDisposed()
        
        var ex2 = Assert.Throws<InvalidOperationException>(() =>
        {
            MulticlassClassificationMetrics metrics =
                mlContext.MulticlassClassification.Evaluate(testOutputDataView,
                    labelColumnName: "LabelKey",
                    predictedLabelColumnName: "PredictedLabel");
        });
        Assert.NotNull(ex2.InnerException);
        Assert.NotNull(ex2.InnerException.InnerException);
        Assert.Equal("Object is disposed.", ex2.InnerException.InnerException.Message); 
    }
    
    [Fact]
    public void DontThrows()
    {
        // throws when transformerShouldResizeToWidthAndHeight != createRandomImagesWithWidthAndHeight
        const int transformerShouldResizeToWidthAndHeight = 2;
        const int createRandomImagesWithWidthAndHeight = 3;
        
        var mlContext = new MLContext();
        var pipeline = CreatePipeline(mlContext, transformerShouldResizeToWidthAndHeight);

        var (trainingInputDatas, trainingInputDatasDataView) = CreateRandomInputData(2, mlContext, createRandomImagesWithWidthAndHeight);
        var transformer = pipeline.Fit(trainingInputDatasDataView);

        var (testInputDatas, testInputDatasDataView) = CreateRandomInputData(2, mlContext, createRandomImagesWithWidthAndHeight);
        var testOutputDataView = transformer.Transform(testInputDatasDataView);

        var h0 = testInputDatas[0].SourceImage.Height;
        
        // I just want to look at the output or output to the log
        var outputDatas = mlContext.Data.CreateEnumerable<OutputData>(testOutputDataView, true).ToArray();

        var h1 = testInputDatas[0].SourceImage.Height;
        
        MulticlassClassificationMetrics metrics =
            mlContext.MulticlassClassification.Evaluate(testOutputDataView,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");
    }
    
    private IEstimator<ITransformer> CreatePipeline(MLContext mlContext, int resizeToWidthAndHeight)
    {
        IEstimator<ITransformer> pipeline =
            mlContext.Transforms.ResizeImages(
                    inputColumnName: nameof(InputData.SourceImage), outputColumnName: "ResizedImage",
                    imageHeight: resizeToWidthAndHeight, imageWidth: resizeToWidthAndHeight)
                .Append(mlContext.Transforms.ExtractPixels(inputColumnName: "ResizedImage",
                    outputColumnName: "ExtractedPixels",
                    interleavePixelColors: true, offsetImage: 177))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: nameof(InputData.LabelValue), outputColumnName: "LabelKey"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                    labelColumnName: "LabelKey", featureColumnName: "ExtractedPixels"
                    /*outputColumnName: "PredictedLabel"*/))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    inputColumnName: "PredictedLabel",
                    outputColumnName: nameof(OutputData.PredictedLabelValue)))
                .AppendCacheCheckpoint(mlContext);

        return pipeline;
    }

    public (InputData[], IDataView) CreateRandomInputData(int imagesCount, MLContext mlContext, int widthAndHeight)
    {
        var random = new Random(1);

        var inputDatas = Enumerable.Range(0, imagesCount).Select(i =>
        {
            var pixelBytes = new byte[widthAndHeight * widthAndHeight * 4];
            random.NextBytes(pixelBytes);
            var image = MLImage.CreateFromPixels(widthAndHeight, widthAndHeight, MLPixelFormat.Rgba32, pixelBytes);
            var inputData = new InputData { SourceImage = image, LabelValue = $"label_{i}" };
            return inputData;
        }).ToArray();

        var inputDatasDataView = mlContext.Data.LoadFromEnumerable(inputDatas);

        return (inputDatas, inputDatasDataView);
    }
}