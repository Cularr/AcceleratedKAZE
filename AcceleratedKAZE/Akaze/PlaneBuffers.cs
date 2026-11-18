using System.Buffers;

namespace AcceleratedKAZE;

public static partial class Akaze
{
    private sealed class PlaneBuffers(bool retain) : IDisposable
    {
        private readonly List<Candidate>?[] candidates = new List<Candidate>[Octaves * Sublevels];
        private readonly Dictionary<(int, int), Grid> grids = [];
        private readonly Dictionary<int, Stack<float[]>> floats = [];
        private readonly Dictionary<int, Stack<bool[]>> flags = [];
        private readonly HashSet<float[]> rented = [];
        private readonly List<bool[]> masks = [];

        public List<Layer> Layers { get; } = new(Octaves * Sublevels);

        public WorkerTeam? Team { get; private set; }

        public List<Candidate> CandidatesFor(int level)
        {
            var list = candidates[level] ??= [];
            list.Clear();
            return list;
        }

        public Grid GridFor(int width, int height)
        {
            if (!grids.TryGetValue((width, height), out var grid))
            {
                grids[(width, height)] = grid = new(width, height, this);
            }

            return grid;
        }

        public void Configure(int workers)
        {
            if (workers == (Team?.Count ?? 1))
            {
                return;
            }

            Team?.Dispose();
            Team = null;
            if (workers > 1)
            {
                Team = new WorkerTeam(workers);
            }
        }

        public float[] Rent(int length)
        {
            var buffer = retain ? Take(floats, length) : ArrayPool<float>.Shared.Rent(length);
            rented.Add(buffer);
            return buffer;
        }

        public bool[] RentMask(int length)
        {
            var mask = retain ? Take(flags, length) : ArrayPool<bool>.Shared.Rent(length);
            mask.AsSpan(0, length).Clear();
            masks.Add(mask);
            return mask;
        }

        public void Release(float[] buffer)
        {
            if (!rented.Remove(buffer))
            {
                throw new InvalidOperationException("Plane was already released.");
            }

            Return(buffer);
        }

        public void ClearCache()
        {
            Team?.Dispose();
            Team = null;
            Layers.Clear();
            Array.Clear(candidates);
            grids.Clear();
            floats.Clear();
            flags.Clear();
        }

        public void Dispose()
        {
            foreach (var buffer in rented)
            {
                Return(buffer);
            }

            rented.Clear();
            foreach (var mask in masks)
            {
                if (retain)
                {
                    flags[mask.Length].Push(mask);
                }
                else
                {
                    ArrayPool<bool>.Shared.Return(mask);
                }
            }

            masks.Clear();
        }

        private static T[] Take<T>(Dictionary<int, Stack<T[]>> cache, int length)
        {
            if (!cache.TryGetValue(length, out var available))
            {
                cache[length] = available = new();
            }

            return available.TryPop(out var buffer) ? buffer : new T[length];
        }

        private void Return(float[] buffer)
        {
            if (retain)
            {
                floats[buffer.Length].Push(buffer);
            }
            else
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
    }
}
