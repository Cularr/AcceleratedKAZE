using AcceleratedKAZE.Tests.Support;

namespace AcceleratedKAZE.Tests;

public class ImageLoaderTests
{
    [Theory]
    [InlineData("color.jpg")]
    [InlineData("color.png")]
    public void Color_decode_matches_OpenCV(string name)
    {
        var image = ImageLoader.ReadGrayscale(Reference.PathOf(name));
        Assert.Equal(161, image.Width);
        Assert.Equal(121, image.Height);
        Reference.AssertSame(Reference.Bytes(name + ".gray"), image.Pixels, image.Width);
    }
}
