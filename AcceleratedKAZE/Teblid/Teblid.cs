using System.Runtime.Intrinsics.X86;

namespace AcceleratedKAZE;

public sealed partial class Teblid : IDisposable
{
    private readonly record struct Learner(int X1, int Y1, int X2, int Y2, int Radius, float Threshold);
    private readonly Learner[] model;
    private readonly float scaleFactor;
    private readonly TeblidFrame frame = new();
    private readonly TestBlock[] blocks;
    private Akaze.WorkerTeam? team;
    public const float AkazeScale = 5f;
    private const int ParallelKeypoints = 64;
    public int DescriptorBytes => model.Length / 8;

    public Teblid(float scaleFactor, int bits = 256)
    {
        if (!float.IsFinite(scaleFactor) || scaleFactor <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(scaleFactor));
        }

        this.scaleFactor = scaleFactor;
        model = bits switch
        {
            256 => Model256,
            512 => Model512,
            _ => throw new ArgumentOutOfRangeException(nameof(bits))
        };
        blocks = bits == 256 ? PackedModels.Short : PackedModels.Long;
    }

    public void Compute(ReadOnlySpan<byte> pixels, int width, int height, ReadOnlySpan<Keypoint> points,
                        Span<byte> destination, int channels = 1)
    {
        ValidateRaw(pixels, width, height, channels, points.Length, destination.Length);
        if (points.IsEmpty)
        {
            return;
        }

        frame.Build(pixels, width, height, channels);
        Compute(frame, points, destination);
    }

    public void Compute(TeblidFrame prepared, ReadOnlySpan<Keypoint> points, Span<byte> destination)
    {
        ValidatePrepared(prepared, points.Length, destination.Length);
        for (int i = 0; i < points.Length; i++)
        {
            Describe(prepared.Table, points[i], prepared.Width, prepared.Height, prepared.Width + 1, destination.Slice(i * DescriptorBytes, DescriptorBytes));
        }
    }

    public void ComputeParallel(ReadOnlySpan<byte> pixels, int width, int height, Keypoint[] points,
                                byte[] destination, int workers, int channels = 1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(workers, 1);
        ValidateRaw(pixels, width, height, channels, points.Length, destination.Length);
        if (points.Length == 0)
        {
            return;
        }

        frame.Build(pixels, width, height, channels);
        ComputeParallel(frame, points, destination, workers);
    }

    public void ComputeParallel(TeblidFrame prepared, Keypoint[] points, byte[] destination, int workers)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(workers, 1);
        if (workers == 1 || points.Length < 2)
        {
            Compute(prepared, points, destination);
            return;
        }
        ValidatePrepared(prepared, points.Length, destination.Length);
        workers = Math.Min(workers, points.Length);
        if (team?.Count != workers)
        {
            Dispose();
            team = new Akaze.WorkerTeam(workers);
        }
        team.Run(new DescribeJob(this, prepared, points, destination), points.Length);
    }

    public void Compute(Akaze.Session session, ReadOnlySpan<byte> pixels, int width, int height, byte[] destination, int channels = 1)
    {
        ArgumentNullException.ThrowIfNull(session);
        ObjectDisposedException.ThrowIf(session.IsDisposed, session);
        ArgumentNullException.ThrowIfNull(destination);
        ValidateRaw(pixels, width, height, channels, session.Count, destination.Length);
        if (session.Count == 0)
        {
            return;
        }

        frame.Build(pixels, width, height, channels);
        Compute(frame, session, destination);
    }

    public void Compute(TeblidFrame prepared, Akaze.Session session, byte[] destination)
    {
        ArgumentNullException.ThrowIfNull(session);
        ObjectDisposedException.ThrowIf(session.IsDisposed, session);
        ArgumentNullException.ThrowIfNull(destination);
        ValidatePrepared(prepared, session.Count, destination.Length);
        var job = new DescribeJob(this, prepared, session.PointBuffer, destination);
        if (session.Team is { } shared && session.Count >= ParallelKeypoints)
        {
            shared.Run(job, session.Count);
        }
        else
        {
            job.Execute(0, session.Count);
        }
    }

    private void ValidateRaw(ReadOnlySpan<byte> pixels, int width, int height, int channels, int count, int length)
    {
        TeblidFrame.Validate(pixels, width, height, channels);
        ValidateDestination(count, length);
    }

    private void ValidatePrepared(TeblidFrame prepared, int count, int length)
    {
        ArgumentNullException.ThrowIfNull(prepared);
        if (prepared.Width == 0)
        {
            throw new InvalidOperationException("Prepare the frame before extracting descriptors.");
        }

        ValidateDestination(count, length);
    }

    private void ValidateDestination(int count, int length)
    {
        if (length < checked(count * DescriptorBytes))
        {
            throw new ArgumentException("Insufficient descriptor storage.", "destination");
        }
    }

    private readonly record struct DescribeJob(Teblid Extractor, TeblidFrame Frame, Keypoint[] Points, byte[] Destination) : Akaze.IRangeJob
    {
        public void Execute(int start, int end)
        {
            int bytes = Extractor.DescriptorBytes;
            for (int i = start; i < end; i++)
            {
                Extractor.Describe(Frame.Table, Points[i], Frame.Width, Frame.Height, Frame.Width + 1, Destination.AsSpan(i * bytes, bytes));
            }
        }
    }

    public void Dispose()
    {
        team?.Dispose();
        team = null;
    }

    private void Describe(int[] integral, Keypoint point, int width, int height, int stride, Span<byte> row)
    {
        float scale = scaleFactor * point.Size / 32;
        float margin = 32 * (scaleFactor * point.Size / 64) * 1.75f;
        bool border = point.X < margin || point.X + margin >= width || point.Y < margin || point.Y + margin >= height;
        double radians = point.Angle * (Math.PI / 180);
        float cos = point.Angle >= 0 ? (float)Math.Cos(radians) : 1;
        float sin = point.Angle >= 0 ? (float)Math.Sin(radians) : 0;
        float a = scale * cos, b = scale * sin;
        float tx = (-a + b) * 16 + point.X, ty = (-b - a) * 16 + point.Y;
        if (Avx2.IsSupported && scale >= 0 && scale < 1_000_000 &&
           float.IsFinite(tx) && float.IsFinite(ty) && Math.Abs(tx) < 1e9f && Math.Abs(ty) < 1e9f)
        {
            DescribeBlocks(integral, a, b, tx, ty, scale, border, width, height, stride, row);
            return;
        }
        int packed = 0;
        for (int i = 0; i < model.Length; i++)
        {
            ref readonly var test = ref model[i];
            int x1 = (int)((a * test.X1 - b * test.Y1 + tx) + .5f);
            int y1 = (int)((b * test.X1 + a * test.Y1 + ty) + .5f);
            int x2 = (int)((a * test.X2 - b * test.Y2 + tx) + .5f);
            int y2 = (int)((b * test.X2 + a * test.Y2 + ty) + .5f);
            int r = (int)(scale * test.Radius + .5f);
            bool set;
            if (border || Math.Min(x1, x2) - r < 0 || Math.Min(y1, y2) - r < 0 ||
                Math.Max(x1, x2) + r >= width || Math.Max(y1, y2) + r >= height)
            {
                float first = ClippedMean(integral, x1, y1, r, width, height, stride);
                float second = ClippedMean(integral, x2, y2, r, width, height, stride);
                set = first - second <= test.Threshold;
            }
            else
            {
                int first = Sum(integral, x1 - r, y1 - r, x1 + r + 1, y1 + r + 1, stride);
                int second = Sum(integral, x2 - r, y2 - r, x2 + r + 1, y2 + r + 1, stride);
                int side = 2 * r + 1;
                set = unchecked(first - second) <= test.Threshold * (side * side);
            }
            packed = (packed << 1) | (set ? 1 : 0);
            if ((i & 7) == 7) { row[i / 8] = (byte)packed; packed = 0; }
        }
    }

    private static int Sum(int[] integral, int left, int top, int right, int bottom, int stride)
    {
        top *= stride;
        bottom *= stride;
        return unchecked(integral[top + left] + integral[bottom + right] - integral[top + right] - integral[bottom + left]);
    }

    private static float ClippedMean(int[] integral, int x, int y, int radius, int width, int height, int stride)
    {
        int left = Math.Clamp(x - radius, 0, width - 1), right = Math.Clamp(x + radius + 1, 1, width);
        int top = Math.Clamp(y - radius, 0, height - 1), bottom = Math.Clamp(y + radius + 1, 1, height);
        return (float)Sum(integral, left, top, right, bottom, stride) / ((right - left) * (bottom - top));
    }
}
