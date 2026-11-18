using AcceleratedKAZE.Tests.Support;

namespace AcceleratedKAZE.Tests;

public class TeblidTests
{
    public static TheoryData<string, int> Files => new()
    {
        { "synthetic.teblid", 256 }, { "synthetic.teblid", 512 },
        { "box.teblid", 256 }, { "box.teblid", 512 },
        { "graf1.teblid", 256 }, { "graf1.teblid", 512 }
    };

    public static TheoryData<string, int, int> SessionRuns => new()
    {
        { "box", 256, 1 }, { "box", 512, 4 }, { "graf1", 256, 4 }, { "graf1", 512, 1 }
    };

    [Theory]
    [MemberData(nameof(Files))]
    public void Compute_matches_OpenCV(string file, int bits)
    {
        foreach (var item in TeblidCase.Read(file))
        {
            using var teblid = new Teblid(item.Scale, bits);
            var actual = new byte[item.Points.Length * teblid.DescriptorBytes];
            teblid.Compute(item.Pixels, item.Width, item.Height, item.Points, actual, item.Channels);
            Reference.AssertSame(item.Expected(bits), actual, teblid.DescriptorBytes);
        }
    }

    [Theory]
    [MemberData(nameof(Files))]
    public void ComputeParallel_matches_OpenCV(string file, int bits)
    {
        foreach (var item in TeblidCase.Read(file))
        {
            using var teblid = new Teblid(item.Scale, bits);
            var actual = new byte[item.Points.Length * teblid.DescriptorBytes];
            teblid.ComputeParallel(item.Pixels, item.Width, item.Height, item.Points, actual, 4, item.Channels);
            Reference.AssertSame(item.Expected(bits), actual, teblid.DescriptorBytes);
        }
    }

    [Fact]
    public void Prepared_frame_serves_both_models()
    {
        var item = TeblidCase.Read("box.teblid")[0];
        var frame = new TeblidFrame();
        frame.Prepare(item.Pixels, item.Width, item.Height, item.Channels);
        foreach (int bits in new[] { 256, 512 })
        {
            using var teblid = new Teblid(item.Scale, bits);
            var actual = new byte[item.Points.Length * teblid.DescriptorBytes];
            teblid.Compute(frame, item.Points, actual);
            Reference.AssertSame(item.Expected(bits), actual, teblid.DescriptorBytes);
        }
    }

    [Theory]
    [MemberData(nameof(SessionRuns))]
    public void Session_keypoints_give_OpenCV_descriptors(string name, int bits, int workers)
    {
        var item = TeblidCase.Read(name + ".teblid")[0];
        using var session = new Akaze.Session();
        session.Compute(item.Pixels, item.Width, item.Height, workers);
        Assert.Equal(item.Points.Length, session.Count);
        for (int i = 0; i < session.Count; i++)
        {
            var point = session.Points[i];
            Assert.Equal(item.Points[i], new Keypoint(point.X, point.Y, point.Size, point.Angle, 0, 0, 0));
        }

        using var teblid = new Teblid(Teblid.AkazeScale, bits);
        var actual = new byte[session.Count * teblid.DescriptorBytes];
        teblid.Compute(session, item.Pixels, item.Width, item.Height, actual);
        Reference.AssertSame(item.Expected(bits), actual, teblid.DescriptorBytes);
    }

    [Fact]
    public void Short_destination_is_rejected()
    {
        var item = TeblidCase.Read("box.teblid")[0];
        using var teblid = new Teblid(item.Scale);
        Assert.Throws<ArgumentException>(() => teblid.Compute(item.Pixels, item.Width, item.Height, item.Points, new byte[1]));
    }
}
