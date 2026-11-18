using AcceleratedKAZE.Tests.Support;

namespace AcceleratedKAZE.Tests;

public class AkazeTests
{
    private static readonly string[] Names =
    [
        "box", "graf1", "graf1-797x637", "noise-257x193", "blank",
        "narrow-200x60", "narrow-200x59", "narrow-200x58", "narrow-60x200", "narrow-59x200"
    ];

    public static TheoryData<string> Images
    {
        get
        {
            var images = new TheoryData<string>();
            foreach (string name in Names)
            {
                images.Add(name);
            }

            return images;
        }
    }

    public static TheoryData<string, int> SessionRuns
    {
        get
        {
            var runs = new TheoryData<string, int>();
            foreach (string name in Names)
            {
                foreach (int workers in new[] { 1, 3, 10 })
                {
                    runs.Add(name, workers);
                }
            }

            return runs;
        }
    }

    [Theory]
    [MemberData(nameof(Images))]
    public void DetectAndCompute_matches_OpenCV(string name)
    {
        var image = Reference.Image(name);
        var features = Akaze.DetectAndCompute(image.Pixels, image.Width, image.Height);
        Reference.AssertSame(Reference.Bytes(name + ".akaze"), Reference.Pack(features), Reference.RecordBytes);
    }

    [Theory]
    [MemberData(nameof(SessionRuns))]
    public void Session_matches_OpenCV(string name, int workers)
    {
        var image = Reference.Image(name);
        using var session = new Akaze.Session();
        session.Compute(image.Pixels, image.Width, image.Height, workers);
        Reference.AssertSame(Reference.Bytes(name + ".akaze"), Reference.Pack(session), Reference.RecordBytes);
    }

    [Fact]
    public void Session_can_be_reused_across_image_sizes()
    {
        using var session = new Akaze.Session();
        foreach (string name in new[] { "graf1", "box", "blank", "graf1" })
        {
            var image = Reference.Image(name);
            session.Compute(image.Pixels, image.Width, image.Height, 3);
            Reference.AssertSame(Reference.Bytes(name + ".akaze"), Reference.Pack(session), Reference.RecordBytes);
        }
    }

    [Fact]
    public void Warm_session_does_not_allocate()
    {
        var image = Reference.Image("box");
        using var session = new Akaze.Session();
        session.Compute(image.Pixels, image.Width, image.Height);
        session.Compute(image.Pixels, image.Width, image.Height);
        long before = GC.GetAllocatedBytesForCurrentThread();
        session.Compute(image.Pixels, image.Width, image.Height);
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
    }

    [Theory]
    [InlineData(0, 64)]
    [InlineData(64, 63)]
    public void Wrong_pixel_count_is_rejected(int width, int height)
    {
        Assert.Throws<ArgumentException>(() => Akaze.DetectAndCompute(new byte[64 * 64], width, height));
    }
}
