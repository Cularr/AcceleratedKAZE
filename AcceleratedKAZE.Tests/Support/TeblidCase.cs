namespace AcceleratedKAZE.Tests.Support;

public sealed record TeblidCase(byte[] Pixels, int Width, int Height, int Channels, float Scale, Keypoint[] Points, byte[] Short, byte[] Long)
{
    public byte[] Expected(int bits) => bits == 256 ? Short : Long;

    public static List<TeblidCase> Read(string name)
    {
        var cases = new List<TeblidCase>();
        using var reader = new BinaryReader(File.OpenRead(Reference.PathOf(name)));
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            int width = reader.ReadInt32();
            int height = reader.ReadInt32();
            int channels = reader.ReadInt32();
            int count = reader.ReadInt32();
            float scale = reader.ReadSingle();
            var pixels = reader.ReadBytes(width * height * channels);
            var points = new Keypoint[count];
            for (int i = 0; i < count; i++)
            {
                points[i] = new(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), 0, 0, 0);
            }

            cases.Add(new(pixels, width, height, channels, scale, points, reader.ReadBytes(count * 32), reader.ReadBytes(count * 64)));
        }

        Assert.NotEmpty(cases);
        return cases;
    }
}
