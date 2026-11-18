using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AcceleratedKAZE;

public static partial class Akaze
{
    private static int FilterVectors(float[] source, float[] target, float[] kernel, ReadOnlySpan<int> offsets,
        int line, int width, bool horizontal, bool replicate, int start)
    {
        if (!Avx.IsSupported || !Fma.IsSupported)
        {
            return start;
        }

        int radius = kernel.Length / 2;
        int end = horizontal ? Math.Min(width - radius, width - width % 8) : width - width % 8;
        int x = start;
        if (end - x < 8)
        {
            return x;
        }

        ref float src = ref MemoryMarshal.GetArrayDataReference(source);
        ref float dst = ref MemoryMarshal.GetArrayDataReference(target);
        if (!replicate)
        {
            int lo = offsets[0], mid = offsets[radius], hi = offsets[^1];
            var outer = Vector256.Create(kernel[^1]);
            var middle = Vector256.Create(kernel[radius]);
            if (kernel[0] < 0)
            {
                for (; x + 8 <= end; x += 8)
                {
                    var diff = Vector256.LoadUnsafe(ref src, (nuint)(hi + x)) - Vector256.LoadUnsafe(ref src, (nuint)(lo + x));
                    diff.StoreUnsafe(ref dst, (nuint)(line + x));
                }
            }
            else if (horizontal && radius > 2)
            {
                for (; x + 8 <= end; x += 8)
                {
                    var a = Vector256.LoadUnsafe(ref src, (nuint)(lo + x)) * outer;
                    var b = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(mid + x)), middle, a);
                    Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(hi + x)), outer, b).StoreUnsafe(ref dst, (nuint)(line + x));
                }
            }
            else
            {
                for (; x + 8 <= end; x += 8)
                {
                    var pair = Vector256.LoadUnsafe(ref src, (nuint)(lo + x)) + Vector256.LoadUnsafe(ref src, (nuint)(hi + x));
                    var center = Vector256.LoadUnsafe(ref src, (nuint)(mid + x));
                    var sum = horizontal && radius == 1 ? Fma.MultiplyAdd(center, middle, pair * outer)
                        : Fma.MultiplyAdd(pair, outer, center * middle);
                    sum.StoreUnsafe(ref dst, (nuint)(line + x));
                }
            }
            return x;
        }
        if (horizontal && kernel.Length == 9)
        {
            int o = offsets[0];
            Vector256<float> k0 = Vector256.Create(kernel[0]), k1 = Vector256.Create(kernel[1]), k2 = Vector256.Create(kernel[2]),
                k3 = Vector256.Create(kernel[3]), k4 = Vector256.Create(kernel[4]), k5 = Vector256.Create(kernel[5]),
                k6 = Vector256.Create(kernel[6]), k7 = Vector256.Create(kernel[7]), k8 = Vector256.Create(kernel[8]);
            for (; x + 8 <= end; x += 8)
            {
                nuint p = (nuint)(o + x);
                var sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p), k0, Vector256<float>.Zero);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 1), k1, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 2), k2, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 3), k3, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 4), k4, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 5), k5, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 6), k6, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 7), k7, sum);
                Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 8), k8, sum).StoreUnsafe(ref dst, (nuint)(line + x));
            }
        }
        else if (horizontal && kernel.Length == 5)
        {
            int o = offsets[0];
            Vector256<float> k2 = Vector256.Create(kernel[2]), k3 = Vector256.Create(kernel[3]), k4 = Vector256.Create(kernel[4]);
            for (; x + 8 <= end; x += 8)
            {
                nuint p = (nuint)(o + x);
                var pair = Vector256.LoadUnsafe(ref src, p + 1) + Vector256.LoadUnsafe(ref src, p + 3);
                var sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 2), k2, pair * k3);
                Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, p + 4) + Vector256.LoadUnsafe(ref src, p), k4, sum).StoreUnsafe(ref dst, (nuint)(line + x));
            }
        }
        else if (!horizontal && kernel.Length == 5)
        {
            int o0 = offsets[0], o1 = offsets[1], o2 = offsets[2], o3 = offsets[3], o4 = offsets[4];
            Vector256<float> k2 = Vector256.Create(kernel[2]), k3 = Vector256.Create(kernel[3]), k4 = Vector256.Create(kernel[4]);
            for (; x + 8 <= end; x += 8)
            {
                var sum = Vector256.LoadUnsafe(ref src, (nuint)(o2 + x)) * k2;
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(o3 + x)) + Vector256.LoadUnsafe(ref src, (nuint)(o1 + x)), k3, sum);
                Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(o4 + x)) + Vector256.LoadUnsafe(ref src, (nuint)(o0 + x)), k4, sum).StoreUnsafe(ref dst, (nuint)(line + x));
            }
        }
        else if (!horizontal && kernel.Length == 9)
        {
            int o0 = offsets[0], o1 = offsets[1], o2 = offsets[2], o3 = offsets[3], o4 = offsets[4], o5 = offsets[5], o6 = offsets[6], o7 = offsets[7], o8 = offsets[8];
            Vector256<float> k4 = Vector256.Create(kernel[4]), k5 = Vector256.Create(kernel[5]), k6 = Vector256.Create(kernel[6]),
                k7 = Vector256.Create(kernel[7]), k8 = Vector256.Create(kernel[8]);
            for (; x + 8 <= end; x += 8)
            {
                var sum = Vector256.LoadUnsafe(ref src, (nuint)(o4 + x)) * k4;
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(o5 + x)) + Vector256.LoadUnsafe(ref src, (nuint)(o3 + x)), k5, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(o6 + x)) + Vector256.LoadUnsafe(ref src, (nuint)(o2 + x)), k6, sum);
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(o7 + x)) + Vector256.LoadUnsafe(ref src, (nuint)(o1 + x)), k7, sum);
                Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src, (nuint)(o8 + x)) + Vector256.LoadUnsafe(ref src, (nuint)(o0 + x)), k8, sum).StoreUnsafe(ref dst, (nuint)(line + x));
            }
        }
        return x;
    }
}
