using System.Numerics;

namespace AcceleratedKAZE;

public static partial class Akaze
{
    private sealed partial record Grid(int W, int H, PlaneBuffers Planes)
    {
        private static readonly float[] Gaussian5 = Gaussian(1, 5);
        private static readonly float[] Gaussian9 = Gaussian(1.6f, 9);
        private static readonly (float[] Derivative, float[] Average) Scharr = Kernels(1, true);
        private static readonly (float[] Derivative, float[] Average)[] Derivatives = ScaleKernels(6);

        public void Run<T>(T job, int items)
            where T : struct, IRangeJob
        {
            if (Planes.Team is { } team && W * H >= ParallelPixels)
            {
                team.Run(job, items);
            }
            else
            {
                job.Execute(0, items);
            }
        }

        public float[] Blur(float[] src, int length)
        {
            var kernel = length == 5 ? Gaussian5 : Gaussian9;
            return Separable(src, kernel, kernel, true);
        }

        public float Contrast(float[] smooth)
        {
            var (dx, dy) = Gradients(smooth, 1, true);
            float max = 0;
            for (int y = 1; y < H - 1; y++)
            {
                max = Math.Max(max, Magnitude(dx, dy, y * W + 1, y * W + W - 1));
            }

            Planes.Release(dy);
            if (max == 0)
            {
                Planes.Release(dx);
                return .03f;
            }

            Span<int> histogram = stackalloc int[300];
            histogram.Clear();
            for (int y = 1; y < H - 1; y++)
            {
                for (int x = 1; x < W - 1; x++)
                {
                    histogram[(int)(dx[y * W + x] * (299 / max))]++;
                }
            }

            Planes.Release(dx);
            int target = (int)(((W - 2) * (H - 2) - histogram[0]) * .7f);
            for (int k = 1, count = 0; k < 300; k++)
            {
                if (count >= target)
                {
                    return max * k / 300;
                }

                count += histogram[k];
            }

            return .03f;
        }

        public float[] Diffuse(float[] src, float[] smooth, float contrast, float time)
        {
            var flow = Conductivity(smooth, 1 / (contrast * contrast));
            var spare = Planes.Rent(W * H);
            int n = (int)MathF.Ceiling(MathF.Sqrt(12 * time + .25f) - .5f - 1e-8f);
            int prime = n + 1;
            while (!IsPrime(prime))
            {
                prime++;
            }

            float factor = 3 * time / (.25f * (n * (n + 1))) * .125f;
            var result = Planes.Rent(W * H);
            for (int k = 0, done = 0; done < n; k++)
            {
                int index = (k + 1) * (n / 2) % prime - 1;
                if (index < 0 || index >= n)
                {
                    continue;
                }

                done++;
                float cos = MathF.Cos(MathF.PI * (2 * index + 1) * (1f / (4 * n + 2)));
                float step = factor / (cos * cos) * .5f;
                var dst = ((n - done) & 1) == 0 ? result : spare;
                dst[0] = src[0];
                dst[W - 1] = src[W - 1];
                dst[(H - 1) * W] = src[(H - 1) * W];
                dst[H * W - 1] = src[H * W - 1];
                Run(new DiffusionJob(this, src, dst, flow, step), H);
                src = dst;
            }

            Planes.Release(flow);
            Planes.Release(spare);
            return result;
        }

        private static float[] Gaussian(float sigma, int length)
        {
            var weights = new double[length];
            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                weights[i] = Math.Exp(-Math.Pow(i - length / 2, 2) / (2 * sigma * (double)sigma));
                sum += weights[i];
            }

            var kernel = new float[length];
            for (int i = 0; i < length; i++)
            {
                kernel[i] = (float)(weights[i] / sum);
            }

            return kernel;
        }

        private static (float[] Derivative, float[] Average)[] ScaleKernels(int count)
        {
            var kernels = new (float[], float[])[count];
            for (int scale = 0; scale < count; scale++)
            {
                kernels[scale] = Kernels(Math.Max(1, scale), false);
            }

            return kernels;
        }

        private static (float[] Derivative, float[] Average) Kernels(int scale, bool scharr)
        {
            float weight = 10f / 3;
            float norm = 1 / (2 * scale * (weight + 2));
            var derivative = new float[2 * scale + 1];
            var average = new float[derivative.Length];
            derivative[0] = -1;
            derivative[^1] = 1;
            average[0] = average[^1] = scharr ? 3 : norm;
            average[scale] = scharr ? 10 : weight * norm;
            return (derivative, average);
        }

        private static bool IsPrime(int n)
        {
            for (int d = 2; d * d <= n; d++)
            {
                if (n % d == 0)
                {
                    return false;
                }
            }

            return true;
        }

        private static int Border(int i, int n, bool replicate)
        {
            if (replicate)
            {
                return Math.Clamp(i, 0, n - 1);
            }

            return i < 0 ? -i : i >= n ? 2 * n - i - 2 : i;
        }

        private void DiffuseRows(float[] src, float[] dst, float[] flow, float step, int start, int end)
        {
            for (int y = start; y < end; y++)
            {
                for (int x = 0; x < W; x++)
                {
                    if (x == 1 && y > 0 && y < H - 1 && Vector.IsHardwareAccelerated)
                    {
                        int p0 = y * W + x;
                        for (; x + Vector<float>.Count <= W - 1; x += Vector<float>.Count, p0 += Vector<float>.Count)
                        {
                            var center = new Vector<float>(src, p0);
                            var conductivity = new Vector<float>(flow, p0);
                            var right = (conductivity + new Vector<float>(flow, p0 + 1)) * (new Vector<float>(src, p0 + 1) - center);
                            var left = (conductivity + new Vector<float>(flow, p0 - 1)) * (new Vector<float>(src, p0 - 1) - center);
                            var down = (conductivity + new Vector<float>(flow, p0 + W)) * (new Vector<float>(src, p0 + W) - center);
                            var up = (conductivity + new Vector<float>(flow, p0 - W)) * (new Vector<float>(src, p0 - W) - center);
                            (center + (right + left + down + up) * step).CopyTo(dst, p0);
                        }
                    }

                    if ((x == 0 || x == W - 1) && (y == 0 || y == H - 1))
                    {
                        continue;
                    }

                    int p = y * W + x;
                    float flux = 0;
                    if (x < W - 1)
                    {
                        flux += (flow[p] + flow[p + 1]) * (src[p + 1] - src[p]);
                    }

                    if (x > 0)
                    {
                        flux += (flow[p] + flow[p - 1]) * (src[p - 1] - src[p]);
                    }

                    if (y < H - 1)
                    {
                        flux += (flow[p] + flow[p + W]) * (src[p + W] - src[p]);
                    }

                    if (y > 0)
                    {
                        flux += (flow[p] + flow[p - W]) * (src[p - W] - src[p]);
                    }

                    dst[p] = src[p] + step * flux;
                }
            }
        }

        private readonly record struct DiffusionJob(Grid Grid, float[] Src, float[] Dst, float[] Flow, float Step) : IRangeJob
        {
            public void Execute(int start, int end) => Grid.DiffuseRows(Src, Dst, Flow, Step, start, end);
        }
    }
}
