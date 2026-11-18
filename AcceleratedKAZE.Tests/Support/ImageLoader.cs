using BitMiracle.LibJpeg.Classic;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AcceleratedKAZE.Tests.Support;

public enum ImageReadMode
{
    Color,
    Grayscale
}

public sealed record GrayImage(byte[] Pixels, int Width, int Height);

public static class ImageLoader
{
    public static GrayImage ReadGrayscale(string path, ImageReadMode mode = ImageReadMode.Color)
    {
        using var stream = File.OpenRead(path);
        bool isJpeg = stream.ReadByte() == 0xff && stream.ReadByte() == 0xd8;
        stream.Position = 0;
        if (isJpeg)
        {
            return ReadJpeg(stream, mode);
        }

        string format = Image.DetectFormat(stream).Name;
        stream.Position = 0;
        using var image = Image.Load<Rgb24>(stream);
        int width = image.Width;
        var pixels = new byte[checked(width * image.Height)];
        bool gray = mode == ImageReadMode.Grayscale;
        image.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < rows.Height; y++)
            {
                var row = rows.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    var p = row[x];
                    pixels[y * width + x] = gray && format == "PNG" ? (byte)((9797 * p.R + 19234 * p.G + 3737 * p.B) >> 15)
                        : gray && format == "TIFF" ? (byte)((4899 * p.R + 9617 * p.G + 1868 * p.B + 8192) >> 14)
                        : Gray(p.R, p.G, p.B);
                }
            }
        });
        return new(pixels, width, image.Height);
    }

    private static byte Gray(int r, int g, int b) => (byte)((9798 * r + 19235 * g + 3735 * b + 16384) >> 15);

    private static GrayImage ReadJpeg(Stream stream, ImageReadMode mode)
    {
        var jpeg = new jpeg_decompress_struct();
        jpeg.jpeg_stdio_src(stream);
        jpeg.jpeg_read_header(true);
        bool gray = mode == ImageReadMode.Grayscale;
        bool ycc = !gray && jpeg.Jpeg_color_space == J_COLOR_SPACE.JCS_YCbCr;
        jpeg.Out_color_space = gray ? J_COLOR_SPACE.JCS_GRAYSCALE : ycc ? J_COLOR_SPACE.JCS_YCbCr : J_COLOR_SPACE.JCS_RGB;
        if (ycc)
        {
            jpeg.Do_fancy_upsampling = false;
        }

        jpeg.jpeg_start_decompress();
        int width = jpeg.Output_width;
        int height = jpeg.Output_height;
        var pixels = new byte[checked(width * height)];
        byte[] samples = ycc ? new byte[checked(pixels.Length * 3)] : [];
        byte[][] rows = [new byte[checked(width * jpeg.Output_components)]];
        for (int y = 0; y < height; y++)
        {
            if (jpeg.jpeg_read_scanlines(rows, 1) != 1)
            {
                throw new InvalidDataException("Incomplete JPEG scanline.");
            }

            var row = rows[0];
            if (gray)
            {
                row.CopyTo(pixels, y * width);
            }
            else if (ycc)
            {
                row.CopyTo(samples, y * width * 3);
            }
            else
            {
                for (int x = 0; x < width; x++)
                {
                    pixels[y * width + x] = Gray(row[3 * x], row[3 * x + 1], row[3 * x + 2]);
                }
            }
        }

        if (ycc)
        {
            ConvertYcc(jpeg, samples, pixels, width, height);
        }

        jpeg.jpeg_finish_decompress();
        return new(pixels, width, height);
    }

    private static void ConvertYcc(jpeg_decompress_struct jpeg, byte[] samples, byte[] pixels, int width, int height)
    {
        var components = jpeg.Comp_info;
        int hmax = components.Max(c => c.H_samp_factor);
        int vmax = components.Max(c => c.V_samp_factor);
        var plane = new Chroma(samples, width, height);
        int h1 = hmax / components[1].H_samp_factor, v1 = vmax / components[1].V_samp_factor;
        int h2 = hmax / components[2].H_samp_factor, v2 = vmax / components[2].V_samp_factor;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;
                int luma = samples[3 * i];
                int cb = plane.At(x, y, 1, h1, v1) - 128;
                int cr = plane.At(x, y, 2, h2, v2) - 128;
                int r = luma + ((91881 * cr + 32768) >> 16);
                int g = luma + ((-22554 * cb - 46802 * cr + 32768) >> 16);
                int b = luma + ((116130 * cb + 32768) >> 16);
                pixels[i] = Gray(Math.Clamp(r, 0, 255), Math.Clamp(g, 0, 255), Math.Clamp(b, 0, 255));
            }
        }
    }

    private readonly record struct Chroma(byte[] Samples, int Width, int Height)
    {
        public int At(int x, int y, int channel, int h, int v)
        {
            int cx = x / h;
            int cy = y / v;
            int nx = cx + ((x & 1) == 0 ? -1 : 1);
            int ny = cy + ((y & 1) == 0 ? -1 : 1);
            int center = Sample(cx, cy, channel, h, v);
            if (h == 2 && v == 2 && (Width + 1) / 2 > 2)
            {
                int column = 3 * center + Sample(cx, ny, channel, h, v);
                int adjacent = 3 * Sample(nx, cy, channel, h, v) + Sample(nx, ny, channel, h, v);
                return (3 * column + adjacent + ((x & 1) == 0 ? 8 : 7)) >> 4;
            }

            if (h == 2 && v == 1 && (Width + 1) / 2 > 2)
            {
                return (3 * center + Sample(nx, cy, channel, h, v) + ((x & 1) == 0 ? 1 : 2)) >> 2;
            }

            if (h == 1 && v == 2)
            {
                return (3 * center + Sample(cx, ny, channel, h, v) + ((y & 1) == 0 ? 1 : 2)) >> 2;
            }

            return center;
        }

        private int Sample(int i, int j, int channel, int h, int v)
        {
            int row = Math.Clamp(j, 0, (Height - 1) / v) * v;
            int column = Math.Clamp(i, 0, (Width - 1) / h) * h;
            return Samples[3 * (row * Width + column) + channel];
        }
    }
}
