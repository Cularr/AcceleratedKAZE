using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AcceleratedKAZE;

public sealed class TeblidFrame
{
    internal int[] Table { get; private set; } = [];
    public int Width { get; private set; }
    public int Height { get; private set; }

    public void Prepare(ReadOnlySpan<byte> pixels, int width, int height, int channels = 1)
    {
        Validate(pixels, width, height, channels);
        Build(pixels, width, height, channels);
    }

    internal static void Validate(ReadOnlySpan<byte> pixels, int width, int height, int channels)
    {
        if (channels is not (1 or 3 or 4))
        {
            throw new ArgumentOutOfRangeException(nameof(channels));
        }

        if (width <= 0 || height <= 0 || (long)width * height != pixels.Length / channels || pixels.Length % channels != 0)
        {
            throw new ArgumentException("Expected a packed 8-bit image of the specified size.");
        }
    }

    internal void Build(ReadOnlySpan<byte> pixels, int width, int height, int channels)
    {
        int stride = checked(width + 1);
        int capacity = checked(stride * (height + 1));
        if (Table.Length < capacity)
        {
            Table = new int[capacity];
        }

        var integral = Table;
        Width = width;
        Height = height;
        Array.Clear(integral, 0, stride);
        if (channels == 1)
        {
            for (int y = 0; y < height; y++)
            {
                var input = pixels.Slice(y * width, width);
                var previous = integral.AsSpan(y * stride + 1, width);
                var current = integral.AsSpan((y + 1) * stride + 1, width);
                integral[(y + 1) * stride] = 0;
                GrayRow(input, previous, current);
            }
            return;
        }
        int src = 0;
        for (int y = 0; y < height; y++)
        {
            int previous = y * stride, current = previous + stride, sum = 0;
            integral[current] = 0;
            for (int x = 1; x <= width; x++, src += channels)
            {
                sum += Gray(pixels[src + 2], pixels[src + 1], pixels[src]);
                integral[current + x] = unchecked(integral[previous + x] + sum);
            }
        }
    }

    private static byte Gray(int r, int g, int b) => (byte)((9798 * r + 19235 * g + 3735 * b + 16384) >> 15);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GrayRow(ReadOnlySpan<byte> input, ReadOnlySpan<int> previous, Span<int> current)
    {
        int x = 0, sum = 0;
        if (Avx2.IsSupported && input.Length >= 8)
        {
            ref byte src = ref MemoryMarshal.GetReference(input);
            ref int prev = ref MemoryMarshal.GetReference(previous);
            ref int dst = ref MemoryMarshal.GetReference(current);
            var carry = Vector256<int>.Zero;
            var lastLane = Vector256.Create(7);
            for (; x <= input.Length - 8; x += 8)
            {
                var bytes = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref src, x));
                var prefix = Sse41.ConvertToVector128Int16(Vector128.CreateScalar(bytes).AsByte());
                prefix += Sse2.ShiftLeftLogical128BitLane(prefix, 2);
                prefix += Sse2.ShiftLeftLogical128BitLane(prefix, 4);
                prefix += Sse2.ShiftLeftLogical128BitLane(prefix, 8);
                var local = Avx2.ConvertToVector256Int32(prefix);
                var total = local + carry;
                carry += Avx2.PermuteVar8x32(local, lastLane);
                (Vector256.LoadUnsafe(ref prev, (nuint)x) + total).StoreUnsafe(ref dst, (nuint)x);
            }
            sum = carry.GetElement(0);
        }
        for (; x < input.Length; x++)
        {
            sum += input[x];
            current[x] = unchecked(previous[x] + sum);
        }
    }
}
