using System.Runtime.CompilerServices;

namespace AcceleratedKAZE;

public static partial class Akaze
{
    [ThreadStatic] private static float[]? window;
    private static float[] Window(int length)
    {
        var rows = window;
        if (rows == null || rows.Length < length)
        {
            window = rows = new float[length];
        }

        return rows;
    }

    private sealed partial record Grid
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float Tap(float[] src, ReadOnlySpan<int> offsets, int r, int y, int x, int k, bool horizontal, bool replicate)
            => src[horizontal ? y * W + Border(x + k, W, replicate) : offsets[k + r] + x];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Accumulate(bool fused, float a, float b, float c) => fused ? MathF.FusedMultiplyAdd(a, b, c) : a * b + c;

        private void FilterRow(float[] src, float[] dst, int line, float[] kernel, ReadOnlySpan<int> offsets, int y, bool horizontal, bool replicate)
        {
            int r = kernel.Length / 2;
            for (int x = 0; x < W; x++)
            {
                if (x == (horizontal ? r : 0))
                {
                    x = FilterVectors(src, dst, kernel, offsets, line, W, horizontal, replicate, x);
                    if (x == W)
                    {
                        break;
                    }
                }
                bool vectorLane = x < W - W % 8;
                float sum = 0;
                if (horizontal && kernel.Length > 5)
                {
                    for (int k = -r; k <= r; k++)
                    {
                        sum = Accumulate(vectorLane, Tap(src, offsets, r, y, x, k, horizontal, replicate), kernel[k + r], sum);
                    }
                }
                else
                {
                    float center = Tap(src, offsets, r, y, x, 0, horizontal, replicate);
                    sum = center * kernel[r];
                    int first = 1;
                    if (horizontal && vectorLane && kernel[0] >= 0)
                    {
                        sum = Accumulate(vectorLane, center, kernel[r],
                            (Tap(src, offsets, r, y, x, 1, horizontal, replicate) + Tap(src, offsets, r, y, x, -1, horizontal, replicate)) * kernel[r + 1]);
                        first = 2;
                    }
                    for (int k = first; k <= r; k++)
                    {
                        float after = Tap(src, offsets, r, y, x, k, horizontal, replicate), before = Tap(src, offsets, r, y, x, -k, horizontal, replicate);
                        sum = Accumulate(vectorLane, kernel[r + k], kernel[0] < 0 ? after - before : after + before, sum);
                    }
                }
                dst[line + x] = sum;
            }
        }

        private void HorizontalRow(float[] src, float[] dst, int line, float[] kernel, int y, bool replicate)
        {
            Span<int> offsets = stackalloc int[kernel.Length];
            for (int k = 0; k < offsets.Length; k++)
            {
                offsets[k] = y * W + k - kernel.Length / 2;
            }

            FilterRow(src, dst, line, kernel, offsets, y, true, replicate);
        }

        private float[] Separable(float[] src, float[] horizontal, float[] vertical, bool replicate)
        {
            var result = Planes.Rent(W * H);
            Run(new SeparableJob(this, src, result, horizontal, vertical, replicate), H);
            return result;
        }
        private readonly record struct SeparableJob(Grid Grid, float[] Src, float[] Dst, float[] Horizontal, float[] Vertical, bool Replicate) : IRangeJob
        {
            public void Execute(int start, int end) => Grid.SeparableRows(Src, Dst, Horizontal, Vertical, Replicate, start, end);
        }

        private void SeparableRows(float[] src, float[] dst, float[] horizontal, float[] vertical, bool replicate, int start, int end)
        {
            int r = vertical.Length / 2, slots = 2 * r + 1;
            var rows = Window(slots * W);
            Span<int> offsets = stackalloc int[slots];
            int next = Math.Max(0, start - r);
            for (int y = start; y < end; y++)
            {
                for (int last = Math.Min(H - 1, y + r); next <= last; next++)
                {
                    HorizontalRow(src, rows, next % slots * W, horizontal, next, replicate);
                }

                for (int k = -r; k <= r; k++)
                {
                    offsets[k + r] = Border(y + k, H, replicate) % slots * W;
                }

                FilterRow(rows, dst, y * W, vertical, offsets, y, false, replicate);
            }
        }

        public (float[] Dx, float[] Dy) Gradients(float[] src, int scale, bool scharr = false)
        {
            var (deriv, avg) = scharr ? Scharr : Derivatives[scale];
            var dx = Planes.Rent(W * H);
            var dy = Planes.Rent(W * H);
            Run(new GradientJob(this, src, dx, dy, deriv, avg, 0), H);
            return (dx, dy);
        }

        public float[] Conductivity(float[] src, float inv)
        {
            var flow = Planes.Rent(W * H);
            Run(new GradientJob(this, src, flow, null, Scharr.Derivative, Scharr.Average, inv), H);
            return flow;
        }
        private readonly record struct GradientJob(Grid Grid, float[] Src, float[] Dx, float[]? Dy, float[] Deriv, float[] Avg, float Inv) : IRangeJob
        {
            public void Execute(int start, int end) => Grid.GradientRows(Src, Dx, Dy, Deriv, Avg, Inv, start, end);
        }

        private void GradientRows(float[] src, float[] dx, float[]? dy, float[] deriv, float[] avg, float inv, int start, int end)
        {
            int r = deriv.Length / 2, slots = 2 * r + 1, span = slots * W;
            var rows = Window(2 * span + 2 * W);
            Span<int> across = stackalloc int[slots];
            Span<int> along = stackalloc int[slots];
            int next = Math.Max(0, start - r);
            for (int y = start; y < end; y++)
            {
                for (int last = Math.Min(H - 1, y + r); next <= last; next++)
                {
                    HorizontalRow(src, rows, next % slots * W, deriv, next, false);
                    HorizontalRow(src, rows, span + next % slots * W, avg, next, false);
                }
                for (int k = -r; k <= r; k++)
                {
                    across[k + r] = Border(y + k, H, false) % slots * W;
                    along[k + r] = span + across[k + r];
                }
                if (dy != null)
                {
                    FilterRow(rows, dx, y * W, avg, across, y, false, false);
                    FilterRow(rows, dy, y * W, deriv, along, y, false, false);
                    continue;
                }
                FilterRow(rows, rows, 2 * span, avg, across, y, false, false);
                FilterRow(rows, rows, 2 * span + W, deriv, along, y, false, false);
                Conduct(rows, 2 * span, 2 * span + W, dx, y * W, W, inv);
            }
        }

        public float[] Response(float[] dx, float[] dy, int scale)
        {
            var response = Planes.Rent(W * H);
            Run(new ResponseJob(this, dx, dy, response, scale), H);
            return response;
        }
        private readonly record struct ResponseJob(Grid Grid, float[] Dx, float[] Dy, float[] Dst, int Scale) : IRangeJob
        {
            public void Execute(int start, int end) => Grid.ResponseRows(Dx, Dy, Dst, Scale, start, end);
        }

        private void ResponseRows(float[] dx, float[] dy, float[] dst, int scale, int start, int end)
        {
            var (deriv, avg) = Derivatives[scale];
            int r = deriv.Length / 2, slots = 2 * r + 1, span = slots * W, row = 3 * span;
            var rows = Window(row + 3 * W);
            Span<int> xx = stackalloc int[slots];
            Span<int> xy = stackalloc int[slots];
            Span<int> yy = stackalloc int[slots];
            int next = Math.Max(0, start - r);
            for (int y = start; y < end; y++)
            {
                for (int last = Math.Min(H - 1, y + r); next <= last; next++)
                {
                    HorizontalRow(dx, rows, next % slots * W, deriv, next, false);
                    HorizontalRow(dx, rows, span + next % slots * W, avg, next, false);
                    HorizontalRow(dy, rows, 2 * span + next % slots * W, avg, next, false);
                }
                for (int k = -r; k <= r; k++)
                {
                    xx[k + r] = Border(y + k, H, false) % slots * W;
                    xy[k + r] = span + xx[k + r];
                    yy[k + r] = 2 * span + xx[k + r];
                }
                FilterRow(rows, rows, row, avg, xx, y, false, false);
                FilterRow(rows, rows, row + W, deriv, xy, y, false, false);
                FilterRow(rows, rows, row + 2 * W, deriv, yy, y, false, false);
                Determinant(rows, row, row + 2 * W, row + W, dst, y * W, W, scale * scale * scale * scale);
            }
        }
    }
}
