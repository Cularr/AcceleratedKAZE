namespace AcceleratedKAZE;

public static partial class Akaze
{
    internal interface IRangeJob
    {
        void Execute(int start, int end);
    }

    internal sealed class WorkerTeam : IDisposable
    {
        private readonly Thread[] threads;
        private readonly AutoResetEvent[] ready;
        private readonly ManualResetEventSlim finished = new(false);
        private readonly Dictionary<Type, object> slots = [];
        private volatile bool stopping;
        private int length;
        private int pending;
        private Action<int, int>? execute;
        private Exception? error;

        public WorkerTeam(int count)
        {
            threads = new Thread[count - 1];
            ready = new AutoResetEvent[threads.Length];
            try
            {
                for (int i = 0; i < threads.Length; i++)
                {
                    int index = i;
                    ready[i] = new(false);
                    var thread = new Thread(() => Work(index)) { IsBackground = true, Name = "AKAZE worker" };
                    thread.Start();
                    threads[i] = thread;
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public int Count => threads.Length + 1;

        public void Run<T>(T job, int items)
            where T : struct, IRangeJob
        {
            lock (slots)
            {
                if (!slots.TryGetValue(typeof(T), out var cached))
                {
                    slots[typeof(T)] = cached = new Slot<T>();
                }

                var slot = (Slot<T>)cached;
                slot.Job = job;
                length = items;
                execute = slot.Run;
                error = null;
                pending = threads.Length;
                finished.Reset();
                foreach (var signal in ready)
                {
                    signal.Set();
                }

                try
                {
                    job.Execute(0, items / Count);
                }
                finally
                {
                    finished.Wait();
                    slot.Job = default;
                }

                if (error != null)
                {
                    throw new InvalidOperationException("AKAZE worker failed", error);
                }
            }
        }

        public void Dispose()
        {
            stopping = true;
            foreach (var signal in ready)
            {
                signal?.Set();
            }

            foreach (var thread in threads)
            {
                thread?.Join();
            }

            foreach (var signal in ready)
            {
                signal?.Dispose();
            }

            finished.Dispose();
        }

        private void Work(int index)
        {
            while (true)
            {
                ready[index].WaitOne();
                if (stopping)
                {
                    return;
                }

                try
                {
                    execute!((int)((long)(index + 1) * length / Count), (int)((long)(index + 2) * length / Count));
                }
                catch (Exception ex)
                {
                    Interlocked.CompareExchange(ref error, ex, null);
                }
                finally
                {
                    if (Interlocked.Decrement(ref pending) == 0)
                    {
                        finished.Set();
                    }
                }
            }
        }

        private sealed class Slot<T>
            where T : struct, IRangeJob
        {
            public Slot()
            {
                Run = Execute;
            }

            public T Job;

            public Action<int, int> Run { get; }

            private void Execute(int start, int end) => Job.Execute(start, end);
        }
    }
}
