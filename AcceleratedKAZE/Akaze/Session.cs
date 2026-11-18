namespace AcceleratedKAZE;

public static partial class Akaze
{
    public const int DescriptorBytes = 61;

    public sealed class Session : IDisposable
    {
        private readonly PlaneBuffers planes = new(true);
        private Keypoint[] points = [];
        private byte[] descriptors = [];
        private (int Width, int Height) shape;
        private bool disposed;

        public int Count { get; private set; }

        public ReadOnlySpan<Keypoint> Points => points.AsSpan(0, Count);

        public ReadOnlySpan<byte> Descriptors => descriptors.AsSpan(0, Count * DescriptorBytes);

        internal Keypoint[] PointBuffer => points;

        internal WorkerTeam? Team => planes.Team;

        internal bool IsDisposed => disposed;

        public void Compute(ReadOnlySpan<byte> pixels, int width, int height, int workers = 1, float threshold = .001f)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (shape != (width, height))
            {
                planes.ClearCache();
                shape = (width, height);
            }

            Count = 0;
            try
            {
                ComputeCore(pixels, width, height, threshold, workers, planes, null, this);
            }
            catch
            {
                Count = 0;
                throw;
            }
            finally
            {
                planes.Dispose();
            }
        }

        public void Dispose()
        {
            disposed = true;
            planes.ClearCache();
            points = [];
            descriptors = [];
            Count = 0;
        }

        internal ref Keypoint PointAt(int i) => ref points[i];

        internal Span<byte> DescriptorAt(int i) => descriptors.AsSpan(DescriptorBytes * i, DescriptorBytes);

        internal void Append(Keypoint point)
        {
            if (Count == points.Length)
            {
                int capacity = Math.Max(64, checked(Count * 2));
                var grownPoints = new Keypoint[capacity];
                var grownDescriptors = new byte[checked(capacity * DescriptorBytes)];
                points.AsSpan(0, Count).CopyTo(grownPoints);
                descriptors.AsSpan(0, Count * DescriptorBytes).CopyTo(grownDescriptors);
                points = grownPoints;
                descriptors = grownDescriptors;
            }

            points[Count++] = point;
        }
    }
}
