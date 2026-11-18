namespace AcceleratedKAZE;

public static partial class Akaze
{
    private sealed partial record Grid
    {
        private readonly record struct AreaTable(int[] Ranges, float[] Weights, int Stride);
        private AreaTable? columns, rows;

        private static AreaTable Area(int source, int target)
        {
            double scale = (double)source / target;
            int stride = (int)Math.Ceiling(scale) + 1;
            var ranges = new int[2 * target];
            var weights = new float[stride * target];
            for (int x = 0; x < target; x++)
            {
                int first = (int)(x * scale), last = (int)Math.Min(source, Math.Ceiling((x + 1) * scale));
                if (last - first > stride)
                {
                    throw new InvalidOperationException("Area span exceeds the weight table.");
                }

                ranges[2 * x] = first;
                ranges[2 * x + 1] = last;
                for (int i = first; i < last; i++)
                {
                    weights[x * stride + i - first] = (float)((Math.Min(i + 1, (x + 1) * scale) - Math.Max(i, x * scale)) / scale);
                }
            }
            return new(ranges, weights, stride);
        }

        public float[] Half(float[] src)
        {
            int w = W / 2, h = H / 2;
            var dst = Planes.Rent(w * h);
            if (W != 2 * w || H != 2 * h)
            {
                columns ??= Area(W, w);
                rows ??= Area(H, h);
            }
            Run(new HalfJob(this, src, dst), h);
            return dst;
        }
        private readonly record struct HalfJob(Grid Grid, float[] Src, float[] Dst) : IRangeJob
        {
            public void Execute(int start, int end) => Grid.HalfRows(Src, Dst, start, end);
        }

        private void HalfRows(float[] src, float[] dst, int start, int end)
        {
            int w = W / 2, h = H / 2;
            if (W == 2 * w && H == 2 * h)
            {
                for (int y = start; y < end; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int p = 2 * y * W + 2 * x;
                        float top = src[p] + src[p + 1];
                        dst[y * w + x] = (x < w - w % 4 ? top + (src[p + W] + src[p + W + 1]) : top + src[p + W] + src[p + W + 1]) * .25f;
                    }
                }

                return;
            }
            var (xRanges, xWeights, xStride) = columns!.Value;
            var (yRanges, yWeights, yStride) = rows!.Value;
            for (int y = start; y < end; y++)
            {
                var line = dst.AsSpan(y * w, w);
                line.Clear();
                int top = yRanges[2 * y];
                for (int j = top; j < yRanges[2 * y + 1]; j++)
                {
                    float weight = yWeights[y * yStride + j - top];
                    var source = src.AsSpan(j * W, W);
                    for (int x = 0; x < w; x++)
                    {
                        int first = xRanges[2 * x], last = xRanges[2 * x + 1];
                        float row = 0;
                        for (int i = first; i < last; i++)
                        {
                            row += source[i] * xWeights[x * xStride + i - first];
                        }

                        line[x] += row * weight;
                    }
                }
            }
        }
    }
}
