using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace AcceleratedKAZE;

public static partial class Akaze
{
    private static void ToUnit(ReadOnlySpan<byte> source, Span<float> target)
    {
        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            ref byte src = ref MemoryMarshal.GetReference(source);
            ref float dst = ref MemoryMarshal.GetReference(target);
            var scale = Vector128.Create(1f / 255);
            for (; i + 16 <= source.Length; i += 16)
            {
                var (low, high) = Vector128.Widen(Vector128.LoadUnsafe(ref src, (nuint)i));
                var (a, b) = Vector128.Widen(low);
                var (c, d) = Vector128.Widen(high);
                (Vector128.ConvertToSingle(a.AsInt32()) * scale).StoreUnsafe(ref dst, (nuint)i);
                (Vector128.ConvertToSingle(b.AsInt32()) * scale).StoreUnsafe(ref dst, (nuint)(i + 4));
                (Vector128.ConvertToSingle(c.AsInt32()) * scale).StoreUnsafe(ref dst, (nuint)(i + 8));
                (Vector128.ConvertToSingle(d.AsInt32()) * scale).StoreUnsafe(ref dst, (nuint)(i + 12));
            }
        }
        for (; i < source.Length; i++)
        {
            target[i] = source[i] * (1f / 255);
        }
    }

    private static void Conduct(float[] rows, int gx, int gy, float[] flow, int line, int count, float inv)
    {
        int i = 0;
        var one = Vector<float>.One;
        var scale = new Vector<float>(inv);
        for (; i + Vector<float>.Count <= count; i += Vector<float>.Count)
        {
            var a = new Vector<float>(rows, gx + i);
            var b = new Vector<float>(rows, gy + i);
            (one / (one + (a * a + b * b) * scale)).CopyTo(flow, line + i);
        }
        for (; i < count; i++)
        {
            flow[line + i] = 1 / (1 + (rows[gx + i] * rows[gx + i] + rows[gy + i] * rows[gy + i]) * inv);
        }
    }

    private static void Determinant(float[] rows, int xx, int yy, int xy, float[] response, int line, int count, float scale)
    {
        int i = 0;
        var factor = new Vector<float>(scale);
        for (; i + Vector<float>.Count <= count; i += Vector<float>.Count)
        {
            var mixed = new Vector<float>(rows, xy + i);
            ((new Vector<float>(rows, xx + i) * new Vector<float>(rows, yy + i) - mixed * mixed) * factor).CopyTo(response, line + i);
        }
        for (; i < count; i++)
        {
            response[line + i] = (rows[xx + i] * rows[yy + i] - rows[xy + i] * rows[xy + i]) * scale;
        }
    }

    private static float Magnitude(float[] dx, float[] dy, int start, int end)
    {
        int i = start;
        var peaks = Vector<float>.Zero;
        for (; i + Vector<float>.Count <= end; i += Vector<float>.Count)
        {
            var a = new Vector<float>(dx, i);
            var b = new Vector<float>(dy, i);
            var length = Vector.SquareRoot(a * a + b * b);
            length.CopyTo(dx, i);
            peaks = Vector.Max(peaks, length);
        }
        float max = 0;
        for (int lane = 0; lane < Vector<float>.Count; lane++)
        {
            max = Math.Max(max, peaks[lane]);
        }

        for (; i < end; i++)
        {
            dx[i] = MathF.Sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
            max = Math.Max(max, dx[i]);
        }
        return max;
    }
}
