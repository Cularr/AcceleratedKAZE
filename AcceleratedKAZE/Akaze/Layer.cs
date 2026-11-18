using System.Numerics;
namespace AcceleratedKAZE;

public static partial class Akaze
{
    private const int ParallelKeypoints = 64;

    private readonly record struct Candidate(int Pixel, float Response, float OffsetX, float OffsetY);

    private readonly struct Layer
    {
        private readonly Grid grid;
        private readonly float[] image;
        private readonly float[] dx;
        private readonly float[] dy;
        private readonly List<Candidate> candidates;
        private readonly bool[] maxima;
        private readonly int octave;
        private readonly int ratio;
        private readonly int scale;
        private readonly float radius;

        public Layer(Grid grid, float[] image, float[] smooth, int octave, float radius, float threshold)
        {
            this.grid = grid;
            this.image = image;
            this.octave = octave;
            this.radius = radius;
            ratio = 1 << octave;
            scale = Round(radius / ratio);
            (dx, dy) = grid.Gradients(smooth, scale);
            var response = grid.Response(dx, dy, scale);
            int length = grid.W * grid.H;
            candidates = grid.Planes.CandidatesFor(grid.Planes.Layers.Count);
            maxima = grid.Planes.RentMask(length);
            int border = Round(10 * MathF.Sqrt(2) * scale) + 1;
            var limit = new Vector<float>(threshold);
            int right = grid.W - border;
            for (int y = border; y < grid.H - border; y++)
            {
                for (int x = border; x < right;)
                {
                    if (x + Vector<float>.Count <= right && Vector.LessThanOrEqualAll(new Vector<float>(response, y * grid.W + x), limit))
                    {
                        x += Vector<float>.Count;
                        continue;
                    }
                    for (int stop = Math.Min(right, x + Vector<float>.Count); x < stop; x++)
                    {
                        int p = y * grid.W + x;
                        if (response[p] <= threshold)
                        {
                            continue;
                        }

                        bool peak = true;
                        for (int j = -1; j <= 1; j++)
                        {
                            for (int k = -1; k <= 1; k++)
                            {
                                if ((j != 0 || k != 0) && response[p] <= response[p + j * grid.W + k])
                                {
                                    peak = false;
                                }
                            }
                        }

                        if (!peak)
                        {
                            continue;
                        }

                        int q = Neighbor(x, y, scale);
                        if (q >= 0 && response[p] <= response[q])
                        {
                            continue;
                        }

                        if (q >= 0)
                        {
                            maxima[q] = false;
                        }

                        maxima[p] = true;
                        candidates.Add(Refine(response, p, grid.W));
                    }
                }
            }

            grid.Planes.Release(response);
        }

        private static Candidate Refine(float[] a, int p, int w)
        {
            float bx = -.5f * (a[p + 1] - a[p - 1]), by = -.5f * (a[p + w] - a[p - w]);
            float xx = a[p + 1] + a[p - 1] - 2 * a[p], yy = a[p + w] + a[p - w] - 2 * a[p];
            float xy = .25f * (a[p + w + 1] + a[p - w - 1] - a[p - w + 1] - a[p + w - 1]);
            double det = (double)xx * yy - (double)xy * xy;
            float ox = det == 0 ? 0 : (float)(((double)bx * yy - (double)by * xy) / det);
            float oy = det == 0 ? 0 : (float)(((double)xx * by - (double)xy * bx) / det);
            return new Candidate(p, a[p], ox, oy);
        }

        private float ResponseAt(int pixel)
        {
            int lo = 0, hi = candidates.Count - 1;
            while (lo < hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                if (candidates[mid].Pixel < pixel)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                }
            }
            return candidates[lo].Response;
        }
        private int Neighbor(int x, int y, int r)
        {
            for (int j = -r; j < r; j++)
            {
                for (int k = -r; k < r; k++)
                {
                    if (j * j + k * k <= r * r && maxima[(y + j) * grid.W + x + k])
                    {
                        return (y + j) * grid.W + x + k;
                    }
                }
            }

            return -1;
        }
        public void Suppress(Layer other)
        {
            int r = Math.Max(scale * ratio / other.ratio, other.scale);
            if (other.ratio == ratio)
            {
                r = radius > other.radius ? scale : other.scale;
            }

            foreach (var candidate in candidates)
            {
                int p = candidate.Pixel;
                if (!maxima[p])
                {
                    continue;
                }

                int q = other.Neighbor((p % grid.W) * ratio / other.ratio, (p / grid.W) * ratio / other.ratio, r);
                if (q >= 0 && candidate.Response > other.ResponseAt(q))
                {
                    other.maxima[q] = false;
                }
            }
        }
        public void Collect(int level, List<Feature>? output, Session? session)
        {
            int first = session?.Count ?? 0;
            int w = grid.W;
            foreach (var candidate in candidates)
            {
                int p = candidate.Pixel;
                if (!maxima[p])
                {
                    continue;
                }

                float ox = candidate.OffsetX, oy = candidate.OffsetY;
                if (Math.Abs(ox) > 1 || Math.Abs(oy) > 1)
                {
                    continue;
                }

                float x = (p % w) * ratio + (ox * ratio + .5f * (ratio - 1));
                float y = (p / w) * ratio + (oy * ratio + .5f * (ratio - 1));
                if (session != null)
                {
                    session.Append(new(x, y, radius * 2, 0, candidate.Response, octave, level));
                }
                else
                {
                    float angle = Orientation(Round(x / ratio), Round(y / ratio));
                    byte[] descriptor = new byte[DescriptorBytes];
                    Describe(x / ratio, y / ratio, angle, descriptor);
                    output!.Add(new(x, y, radius * 2, angle, candidate.Response, octave, level, descriptor));
                }
            }
            if (session != null)
            {
                var job = new DescriptorJob(this, session, first);
                int count = session.Count - first;
                if (grid.Planes.Team is { } team && count >= ParallelKeypoints)
                {
                    team.Run(job, count);
                }
                else
                {
                    job.Execute(0, count);
                }
            }
        }
        private readonly record struct DescriptorJob(Layer Layer, Session Session, int First) : IRangeJob
        {
            public void Execute(int start, int end)
            {
                for (int i = First + start; i < First + end; i++)
                {
                    ref var p = ref Session.PointAt(i);
                    float x = p.X / Layer.ratio, y = p.Y / Layer.ratio;
                    float angle = Layer.Orientation(Round(x), Round(y));
                    p = p with { Angle = angle };
                    Layer.Describe(x, y, angle, Session.DescriptorAt(i));
                }
            }
        }
        private static float Angle(float y, float x)
        {
            const float degrees = (float)(180 / Math.PI);
            float ax = Math.Abs(x), ay = Math.Abs(y);
            float c = Math.Min(ax, ay) / (Math.Max(ax, ay) + (float)2.220446049250313e-16);
            float s = c * c;
            float a = (((-.04432655554792128f * degrees * s + .1555786518463281f * degrees) * s
                      - .3258083974640975f * degrees) * s + .9997878412794807f * degrees) * c;
            if (ax < ay)
            {
                a = 90 - a;
            }

            if (x < 0)
            {
                a = 180 - a;
            }

            return y < 0 ? 360 - a : a;
        }

        private static readonly int[] CellSteps = [10, 7, 5];

        private static readonly float[,] Weights =
        {
            { 0.02546481f, 0.02350698f, 0.01849125f, 0.01239505f, 0.00708017f, 0.00344629f, 0.00142946f },
            { 0.02350698f, 0.02169968f, 0.01706957f, 0.01144208f, 0.00653582f, 0.00318132f, 0.00131956f },
            { 0.01849125f, 0.01706957f, 0.01342740f, 0.00900066f, 0.00514126f, 0.00250252f, 0.00103800f },
            { 0.01239505f, 0.01144208f, 0.00900066f, 0.00603332f, 0.00344629f, 0.00167749f, 0.00069579f },
            { 0.00708017f, 0.00653582f, 0.00514126f, 0.00344629f, 0.00196855f, 0.00095820f, 0.00039744f },
            { 0.00344629f, 0.00318132f, 0.00250252f, 0.00167749f, 0.00095820f, 0.00046640f, 0.00019346f },
            { 0.00142946f, 0.00131956f, 0.00103800f, 0.00069579f, 0.00039744f, 0.00019346f, 0.00008024f }
        };

        private float Orientation(int x, int y)
        {
            Span<float> gradientX = stackalloc float[109];
            Span<float> gradientY = stackalloc float[109];
            Span<int> bins = stackalloc int[109];
            Span<int> order = stackalloc int[109];
            Span<int> starts = stackalloc int[43];
            starts.Clear();
            int index = 0;
            for (int j = -5; j <= 5; j++)
            {
                for (int k = -5; k <= 5; k++)
                {
                    if (j * j + k * k >= 36)
                    {
                        continue;
                    }

                    int p = (y + j * scale) * grid.W + x + k * scale;
                    float w = Weights[Math.Abs(j), Math.Abs(k)], a = w * dx[p], b = w * dy[p];
                    int bin = (int)(Angle(b, a) * (float)(Math.PI / 180) / (float)(2 * Math.PI / 42)) % 42;
                    gradientX[index] = a;
                    gradientY[index] = b;
                    bins[index++] = bin;
                    starts[bin]++;
                }
            }

            for (int bin = 1; bin <= 42; bin++)
            {
                starts[bin] += starts[bin - 1];
            }

            for (int i = 0; i < index; i++)
            {
                order[--starts[bins[i]]] = i;
            }

            float best = -1, bx = 0, by = 0;
            for (int start = 0; start < 42; start++)
            {
                float a = 0, b = 0;
                for (int j = 0; j < 7; j++)
                {
                    int bin = (start + j) % 42;
                    for (int i = starts[bin]; i < starts[bin + 1]; i++)
                    {
                        a += gradientX[order[i]];
                        b += gradientY[order[i]];
                    }
                }
                float norm = a * a + b * b;
                if (norm > best) { best = norm; bx = a; by = b; }
            }
            return Angle(by, bx);
        }
        private readonly record struct DescriptorSample(float Intensity, float Dx, float Dy, int Valid);

        private void Describe(float x, float y, float angle, Span<byte> descriptor)
        {
            descriptor.Clear();
            float radians = angle * (float)(Math.PI / 180), co = MathF.Cos(radians), si = MathF.Sin(radians);
            Span<DescriptorSample> samples = stackalloc DescriptorSample[21 * 21];
            for (int k = -10; k <= 10; k++)
            {
                for (int l = -10; l <= 10; l++)
                {
                    int px = Round(x + (-l * si * scale + k * co * scale)), py = Round(y + (l * co * scale + k * si * scale));
                    int index = (k + 10) * 21 + l + 10;
                    if ((uint)px >= grid.W || (uint)py >= grid.H) { samples[index] = default; continue; }
                    int p = py * grid.W + px;
                    samples[index] = new(image[p], -dx[p] * si + dy[p] * co, dx[p] * co + dy[p] * si, 1);
                }
            }

            int bit = 0;
            Span<float> values = stackalloc float[48];
            foreach (int step in CellSteps)
            {
                int cells = 0;
                for (int i = -10; i < 10; i += step)
                {
                    for (int j = -10; j < 10; j += step)
                    {
                        var cell = values.Slice(3 * cells++, 3);
                        float intensity = 0, gx = 0, gy = 0;
                        int count = 0;
                        for (int k = i; k < i + step; k++)
                        {
                            for (int l = j; l < j + step; l++)
                            {
                                var sample = samples[(k + 10) * 21 + l + 10];
                                intensity += sample.Intensity;
                                gx += sample.Dx;
                                gy += sample.Dy;
                                count += sample.Valid;
                            }
                        }

                        float inv = count == 0 ? 0 : 1f / count;
                        cell[0] = intensity * inv;
                        cell[1] = gx * inv;
                        cell[2] = gy * inv;
                    }
                }

                for (int c = 0; c < 3; c++)
                {
                    for (int i = 0; i < cells; i++)
                    {
                        for (int j = i + 1; j < cells; j++, bit++)
                        {
                            if (values[3 * i + c] > values[3 * j + c])
                            {
                                descriptor[bit / 8] |= (byte)(1 << (bit % 8));
                            }
                        }
                    }
                }
            }

        }
    }
}
