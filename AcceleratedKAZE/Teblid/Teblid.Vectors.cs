using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AcceleratedKAZE;

public sealed partial class Teblid
{
    private readonly record struct TestBlock(Vector256<float> X1, Vector256<float> Y1,
        Vector256<float> X2, Vector256<float> Y2, Vector256<float> Radius, Vector256<float> Threshold);

    private static class PackedModels
    {
        internal static readonly TestBlock[] Short = Pack(Model256);
        internal static readonly TestBlock[] Long = Pack(Model512);
        private static TestBlock[] Pack(Learner[] model)
        {
            foreach (var t in model)
            {
                if (t.X1 is < 0 or > 31 || t.X2 is < 0 or > 31 || t.Y1 is < 0 or > 31 ||
                   t.Y2 is < 0 or > 31 || t.Radius is < 0 or > 7)
                {
                    throw new InvalidOperationException("Learner exceeds the vector kernel bounds.");
                }
            }

            var blocks = new TestBlock[model.Length / 8];
            for (int i = 0; i < blocks.Length; i++)
            {
                Vector256<float> Load(Func<Learner, float> field)
                {
                    var v = Vector256<float>.Zero;
                    for (int lane = 0; lane < 8; lane++)
                    {
                        v = v.WithElement(7 - lane, field(model[8 * i + lane]));
                    }

                    return v;
                }
                blocks[i] = new(Load(t => t.X1), Load(t => t.Y1), Load(t => t.X2), Load(t => t.Y2), Load(t => t.Radius), Load(t => t.Threshold));
            }
            return blocks;
        }
    }

    private static bool AxisInside(float u, float v, float translation, int radius, int limit)
    {
        float endU = u * 31, endV = v * 31;
        int low = (int)((Math.Min(0, endU) + Math.Min(0, endV) + translation) + .5f);
        int high = (int)((Math.Max(0, endU) + Math.Max(0, endV) + translation) + .5f);
        return low - radius >= 0 && high + radius < limit;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> InteriorSum(int* image, Vector256<int> start, Vector256<int> side, Vector256<int> down)
    {
        var bottom = start + down;
        return Avx2.GatherVector256(image, start, 4) - Avx2.GatherVector256(image, start + side, 4)
             + Avx2.GatherVector256(image, bottom + side, 4) - Avx2.GatherVector256(image, bottom, 4);
    }

    private static Vector256<int> Round(Vector256<float> value) => Avx.ConvertToVector256Int32WithTruncation(value + Vector256.Create(.5f));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> BoxSum(int* image, Vector256<int> left, Vector256<int> right,
        Vector256<int> top, Vector256<int> bottom, Vector256<int> stride)
    {
        top *= stride;
        bottom *= stride;
        return Avx2.GatherVector256(image, top + left, 4) + Avx2.GatherVector256(image, bottom + right, 4)
             - Avx2.GatherVector256(image, top + right, 4) - Avx2.GatherVector256(image, bottom + left, 4);
    }

    private static unsafe Vector256<float> Mean(int* image, Vector256<int> x, Vector256<int> y, Vector256<int> r,
        Vector256<int> width, Vector256<int> height, Vector256<int> stride)
    {
        var one = Vector256.Create(1);
        var left = Vector256.Clamp(x - r, Vector256<int>.Zero, width - one);
        var right = Vector256.Clamp(x + r + one, one, width);
        var top = Vector256.Clamp(y - r, Vector256<int>.Zero, height - one);
        var bottom = Vector256.Clamp(y + r + one, one, height);
        return Avx.ConvertToVector256Single(BoxSum(image, left, right, top, bottom, stride)) /
               Avx.ConvertToVector256Single((right - left) * (bottom - top));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void DescribeBlocks(int[] integral, float a, float b, float tx, float ty, float scale, bool border, int width, int height, int stride, Span<byte> row)
    {
        var va = Vector256.Create(a);
        var vb = Vector256.Create(b);
        var vx = Vector256.Create(tx);
        var vy = Vector256.Create(ty);
        var vw = Vector256.Create(width);
        var vh = Vector256.Create(height);
        var vs = Vector256.Create(stride);
        var one = Vector256.Create(1);
        int maxRadius = (int)(7 * scale + .5f);
        bool interior = !border && AxisInside(a, -b, tx, maxRadius, width) && AxisInside(b, a, ty, maxRadius, height);
        fixed (int* image = integral)
        {
            for (int i = 0; i < blocks.Length; i++)
            {
                ref readonly var t = ref blocks[i];
                var x1 = Round(va * t.X1 - vb * t.Y1 + vx);
                var y1 = Round(vb * t.X1 + va * t.Y1 + vy);
                var x2 = Round(va * t.X2 - vb * t.Y2 + vx);
                var y2 = Round(vb * t.X2 + va * t.Y2 + vy);
                var r = Round(Vector256.Create(scale) * t.Radius);
                if (interior)
                {
                    var span = r + r + one;
                    var down = span * vs;
                    var first = InteriorSum(image, (y1 - r) * vs + (x1 - r), span, down);
                    var second = InteriorSum(image, (y2 - r) * vs + (x2 - r), span, down);
                    row[i] = (byte)Avx.MoveMask(Avx.CompareLessThanOrEqual(Avx.ConvertToVector256Single(first - second),
                        t.Threshold * Avx.ConvertToVector256Single(span * span)));
                    continue;
                }
                var inside = Vector256.GreaterThanOrEqual(Vector256.Min(x1, x2) - r, Vector256<int>.Zero) &
                           Vector256.GreaterThanOrEqual(Vector256.Min(y1, y2) - r, Vector256<int>.Zero) &
                           Vector256.LessThan(Vector256.Max(x1, x2) + r, vw) & Vector256.LessThan(Vector256.Max(y1, y2) + r, vh);
                int mask = Avx.MoveMask(inside.AsSingle());
                if (border || mask != 255)
                {
                    var diff = Mean(image, x1, y1, r, vw, vh, vs) - Mean(image, x2, y2, r, vw, vh, vs);
                    row[i] = (byte)Avx.MoveMask(Avx.CompareLessThanOrEqual(diff, t.Threshold));
                    if (border || mask == 0)
                    {
                        continue;
                    }

                    x1 &= inside;
                    y1 &= inside;
                    x2 &= inside;
                    y2 &= inside;
                    r &= inside;
                }
                var delta = BoxSum(image, x1 - r, x1 + r + one, y1 - r, y1 + r + one, vs) - BoxSum(image, x2 - r, x2 + r + one, y2 - r, y2 + r + one, vs);
                var side = r + r + one;
                int bits = Avx.MoveMask(Avx.CompareLessThanOrEqual(Avx.ConvertToVector256Single(delta), t.Threshold * Avx.ConvertToVector256Single(side * side)));
                row[i] = mask == 255 ? (byte)bits : (byte)((row[i] & ~mask) | (bits & mask));
            }
        }
    }
}
