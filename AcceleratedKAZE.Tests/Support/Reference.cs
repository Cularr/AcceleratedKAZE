using System.Runtime.InteropServices;

namespace AcceleratedKAZE.Tests.Support;

public static class Reference
{
    public const int RecordBytes = 89;

    public static string PathOf(string name) => Path.Combine(AppContext.BaseDirectory, "Reference", name);

    public static byte[] Bytes(string name) => File.ReadAllBytes(PathOf(name));

    public static GrayImage Image(string name)
    {
        switch (name)
        {
            case "blank":
                return new(new byte[128 * 128], 128, 128);
            case "graf1-797x637":
                var full = Image("graf1");
                var crop = new byte[797 * 637];
                for (int y = 0; y < 637; y++)
                {
                    full.Pixels.AsSpan(y * full.Width, 797).CopyTo(crop.AsSpan(y * 797));
                }

                return new(crop, 797, 637);
            default:
                if (!File.Exists(PathOf(name + ".gray")))
                {
                    return ImageLoader.ReadGrayscale(PathOf(name + ".png"), ImageReadMode.Grayscale);
                }

                var size = name[(name.LastIndexOf('-') + 1)..].Split('x');
                return new(Bytes(name + ".gray"), int.Parse(size[0]), int.Parse(size[1]));
        }
    }

    public static byte[] Pack(IReadOnlyList<Feature> features)
    {
        var bytes = new byte[features.Count * RecordBytes];
        for (int i = 0; i < features.Count; i++)
        {
            var f = features[i];
            Write(bytes.AsSpan(i * RecordBytes), new(f.X, f.Y, f.Size, f.Angle, f.Response, f.Octave, f.ClassId), f.Descriptor);
        }

        return bytes;
    }

    public static byte[] Pack(Akaze.Session session)
    {
        var bytes = new byte[session.Count * RecordBytes];
        for (int i = 0; i < session.Count; i++)
        {
            Write(bytes.AsSpan(i * RecordBytes), session.Points[i], session.Descriptors.Slice(61 * i, 61));
        }

        return bytes;
    }

    public static void AssertSame(byte[] expected, ReadOnlySpan<byte> actual, int unit)
    {
        Assert.Equal(expected.Length, actual.Length);
        int difference = expected.AsSpan().CommonPrefixLength(actual);
        Assert.True(difference == expected.Length, $"First difference in item {difference / unit}, byte {difference % unit}.");
    }

    private static void Write(Span<byte> target, Keypoint point, ReadOnlySpan<byte> descriptor)
    {
        MemoryMarshal.Write(target, in point);
        descriptor.CopyTo(target[28..]);
    }
}
