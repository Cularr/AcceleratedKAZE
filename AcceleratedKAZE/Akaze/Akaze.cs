namespace AcceleratedKAZE;

public static partial class Akaze
{
    private const int Octaves = 4;
    private const int Sublevels = 4;
    private const int ParallelPixels = 32768;
    private const int MinimumSide = 59;

    public static IReadOnlyList<Feature> DetectAndCompute(byte[] grayscale, int width, int height, float threshold = .001f)
    {
        ArgumentNullException.ThrowIfNull(grayscale);
        return DetectAndCompute(grayscale.AsSpan(), width, height, threshold);
    }

    public static IReadOnlyList<Feature> DetectAndCompute(ReadOnlySpan<byte> grayscale, int width, int height, float threshold = .001f)
    {
        using var planes = new PlaneBuffers(false);
        var result = new List<Feature>();
        ComputeCore(grayscale, width, height, threshold, 1, planes, result, null);
        return result;
    }

    private static int Round(float value) => (int)MathF.Round(value);

    private static void ComputeCore(ReadOnlySpan<byte> grayscale, int width, int height, float threshold,
        int workers, PlaneBuffers planes, List<Feature>? result, Session? session)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workers);
        if (width <= 0 || height <= 0 || grayscale.Length != (long)width * height)
        {
            throw new ArgumentException("Expected one byte per pixel.");
        }

        if (width < MinimumSide || height < MinimumSide)
        {
            return;
        }

        var input = planes.Rent(grayscale.Length);
        ToUnit(grayscale, input.AsSpan(0, grayscale.Length));
        planes.Configure(workers);
        var image = planes.GridFor(width, height);
        var smooth = image.Blur(input, 5);
        float contrast = image.Contrast(smooth);
        planes.Release(smooth);
        var layers = planes.Layers;
        layers.Clear();
        float[] current = [];
        float previousTime = 0;
        for (int octave = 0; octave < Octaves; octave++)
        {
            int ratio = 1 << octave;
            if (octave > 0)
            {
                if (width / ratio < 80 || height / ratio < 40)
                {
                    break;
                }

                current = image.Half(current);
                image = planes.GridFor(width / ratio, height / ratio);
                contrast *= .75f;
            }

            for (int sub = 0; sub < Sublevels; sub++)
            {
                float sigma = 1.6f * MathF.Pow(2, octave + sub * .25f);
                float time = .5f * (sigma * sigma);
                if (layers.Count == 0)
                {
                    current = smooth = image.Blur(input, 9);
                    planes.Release(input);
                }
                else
                {
                    smooth = image.Blur(current, 5);
                    var previous = current;
                    current = image.Diffuse(previous, smooth, contrast, time - previousTime);
                    if (sub == 0)
                    {
                        planes.Release(previous);
                    }
                }

                layers.Add(new Layer(image, current, smooth, octave, sigma * 1.5f, threshold));
                if (!ReferenceEquals(current, smooth))
                {
                    planes.Release(smooth);
                }

                previousTime = time;
            }
        }

        for (int i = 1; i < layers.Count; i++)
        {
            layers[i].Suppress(layers[i - 1]);
        }

        for (int i = layers.Count - 2; i >= 0; i--)
        {
            layers[i].Suppress(layers[i + 1]);
        }

        for (int i = 0; i < layers.Count; i++)
        {
            layers[i].Collect(i, result, session);
        }
    }
}
